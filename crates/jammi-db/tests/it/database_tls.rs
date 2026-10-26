//! A database source verifies its server against the CA certificate it holds:
//! a server whose certificate that CA signed connects, verified, and a server
//! it did not sign is refused at registration.
//!
//! The source holds the CA as the PEM a remote client sends, never a path on
//! the engine's host. Each test serves the lane's database over TLS with a
//! certificate whose CA it knows, and registers a source holding that CA and
//! one holding an unrelated CA.

use jammi_db::catalog::backend::BackendKind;
use jammi_db::session::JammiSession;
use jammi_db::source::{CaCertificate, SourceDefinition};
use jammi_test_utils::make_test_session;
use rcgen::{BasicConstraints, CertificateParams, DnType, IsCa, KeyPair};
use tempfile::{tempdir, TempDir};

/// A certificate authority minted for one test: its parameters and key,
/// which sign, and its certificate, as PEM.
fn authority(name: &str) -> (CertificateParams, KeyPair, String) {
    let key = KeyPair::generate().unwrap();
    let mut params = CertificateParams::new(Vec::<String>::new()).unwrap();
    params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    params.distinguished_name.push(DnType::CommonName, name);
    let pem = params.self_signed(&key).unwrap().pem();
    (params, key, pem)
}

/// `pem` as the CA certificate a source holds.
fn ca(pem: &str) -> CaCertificate {
    CaCertificate::from_pem(pem).unwrap()
}

/// A session, and the outcome of registering `url` in it as the source `db`
/// holding `ca`.
async fn register(
    url: &str,
    ca: Option<CaCertificate>,
) -> (JammiSession, TempDir, jammi_db::error::Result<()>) {
    let dir = tempdir().unwrap();
    let session = make_test_session(BackendKind::Sqlite, dir.path()).await;
    let definition = SourceDefinition::from_url(url, None).unwrap();
    let mut connection = definition.connection;
    connection.ca_certificate = ca;
    let registered = session
        .add_source("db", definition.source_type, connection)
        .await;
    (session, dir, registered)
}

/// The rows `db.public.t` holds, read through the session.
async fn count(session: &JammiSession) -> i64 {
    let batches = session
        .sql("SELECT COUNT(*) AS n FROM db.public.t")
        .await
        .unwrap();
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap()
        .value(0)
}

#[cfg(feature = "live-postgres-tests")]
mod postgres {
    use super::*;

    use sqlx::postgres::{PgConnectOptions, PgSslMode};
    use sqlx::{ConnectOptions, Connection, PgConnection};

    /// The certificate and key the Postgres image ships. The key is the
    /// packaging's, readable to the server and no one else, as Postgres
    /// requires of a key; the certificate is self-signed for whichever host
    /// built the image, so it verifies as its own CA, by its chain alone.
    const CERTIFICATE: &str = "/etc/ssl/certs/ssl-cert-snakeoil.pem";
    const KEY: &str = "/etc/ssl/private/ssl-cert-snakeoil.key";

    /// Serve the lane's Postgres over TLS with the image's certificate, and
    /// return the certificate once the server answers over TLS. The wait
    /// does not verify: this client's TLS (rustls) takes no self-signed
    /// server certificate as its own root, where the engine's Postgres
    /// driver (OpenSSL) does, and verifying is the engine's to prove.
    async fn serve() -> String {
        let server = jammi_test_utils::postgres_server_url();
        let mut admin = PgConnection::connect(&server).await.unwrap();
        for statement in [
            format!("ALTER SYSTEM SET ssl_cert_file = '{CERTIFICATE}'"),
            format!("ALTER SYSTEM SET ssl_key_file = '{KEY}'"),
            "ALTER SYSTEM SET ssl = on".into(),
            "SELECT pg_reload_conf()".into(),
        ] {
            sqlx::query(&statement).execute(&mut admin).await.unwrap();
        }
        let certificate: String = sqlx::query_scalar("SELECT pg_read_file($1)")
            .bind(CERTIFICATE)
            .fetch_one(&mut admin)
            .await
            .unwrap();
        let over_tls = server
            .parse::<PgConnectOptions>()
            .unwrap()
            .ssl_mode(PgSslMode::Require);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            match over_tls.connect().await {
                Ok(connection) => {
                    connection.close().await.unwrap();
                    return certificate;
                }
                Err(_) if std::time::Instant::now() < deadline => {
                    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                }
                Err(e) => panic!("the server never served its certificate: {e}"),
            }
        }
    }

    #[tokio::test]
    async fn a_postgres_source_verifies_its_server_against_the_ca_it_holds() {
        let certificate = CaCertificate::from_pem(serve().await).unwrap();
        let source_dir = tempdir().unwrap();
        let url = jammi_test_utils::postgres_database_holding(
            source_dir.path(),
            "CREATE TABLE t (id INTEGER); INSERT INTO t VALUES (1), (2);",
        )
        .await;
        // The image's certificate names another host: verified by its chain.
        let mut verified = url::Url::parse(&url).unwrap();
        verified.set_query(Some("sslmode=verify-ca"));
        let verified = verified.to_string();

        let (session, _dir, registered) = register(&verified, Some(certificate)).await;
        registered.expect("the server's certificate is the CA the source holds");
        assert_eq!(count(&session).await, 2);

        let (_, _, unrelated) = authority("an unrelated CA");
        let (_, _dir, refused) = register(&verified, Some(ca(&unrelated))).await;
        refused.expect_err("a CA that did not sign the server's certificate refuses it");
    }
}

#[cfg(feature = "live-mysql-tests")]
mod mysql {
    use super::*;

    use mysql_async::prelude::Queryable;
    use rcgen::Issuer;

    /// Serve the lane's MySQL with a certificate `issuer` signs for the host
    /// the test reaches it by: the issuer's certificate (`ca`), the server's
    /// and its key written into the server's secure-file directory, and the
    /// server's TLS reloaded from them.
    async fn serve(issuer: &Issuer<'_, KeyPair>, ca: &str, host: &str) {
        let key = KeyPair::generate().unwrap();
        let certificate = CertificateParams::new(vec![host.to_string()])
            .unwrap()
            .signed_by(&key, issuer)
            .unwrap();
        let server = jammi_test_resources::env("JAMMI_TEST_MYSQL_URL");
        let mut admin = mysql_async::Conn::from_url(format!(
            "{server}?require_ssl=true&verify_ca=false&verify_identity=false"
        ))
        .await
        .unwrap();
        let dir: String = admin
            .query_first("SELECT @@secure_file_priv")
            .await
            .unwrap()
            .expect("the server names its secure-file directory");
        let stem = uuid::Uuid::new_v4().simple().to_string();
        let mut paths = Vec::new();
        for (name, pem) in [
            ("ca", ca.to_string()),
            ("crt", certificate.pem()),
            ("key", key.serialize_pem()),
        ] {
            let path = format!("{}/jammi-{stem}.{name}", dir.trim_end_matches('/'));
            admin
                .query_drop(format!(
                    "SELECT X'{}' INTO DUMPFILE '{path}'",
                    hex::encode(pem)
                ))
                .await
                .unwrap();
            paths.push(path);
        }
        admin
            .query_drop(format!(
                "SET GLOBAL ssl_ca = '{}'; SET GLOBAL ssl_cert = '{}'; SET GLOBAL ssl_key = '{}'; \
                 ALTER INSTANCE RELOAD TLS",
                paths[0], paths[1], paths[2]
            ))
            .await
            .unwrap();
        admin.disconnect().await.unwrap();
    }

    #[tokio::test]
    async fn a_mysql_source_verifies_its_server_against_the_ca_it_holds() {
        let (params, key, pem) = authority("jammi test mysql CA");
        let url = crate::common::mysql::mysql_database_holding(
            "CREATE TABLE t (id INT); INSERT INTO t VALUES (1), (2);",
        )
        .await;
        // No mode named: the source connects `required`, verifying the
        // certificate and the host it names.
        let mut verified = url::Url::parse(&url).unwrap();
        verified.set_query(None);
        let host = verified.host_str().unwrap().to_string();
        let verified = verified.to_string();
        serve(&Issuer::new(params, key), &pem, &host).await;

        let (session, _dir, registered) = register(&verified, Some(ca(&pem))).await;
        registered.expect("the server's certificate is signed by the CA the source holds");
        assert_eq!(count(&session).await, 2);

        let (_, _dir, unknown) = register(&verified, None).await;
        unknown.expect_err("a CA the engine's host does not trust verifies nothing");
        let (_, _, unrelated) = authority("an unrelated CA");
        let (_, _dir, refused) = register(&verified, Some(ca(&unrelated))).await;
        refused.expect_err("a CA that did not sign the server's certificate refuses it");
    }
}
