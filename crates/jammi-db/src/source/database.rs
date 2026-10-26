//! What the database sources share: the URL a database source connects
//! through.

use crate::error::{JammiError, Result};
use crate::source::SourceConnection;

/// The connection URL of a database source, which it cannot connect without.
pub(crate) fn url<'a>(source_id: &str, connection: &'a SourceConnection) -> Result<&'a str> {
    connection.url.as_deref().ok_or_else(|| JammiError::Source {
        source_id: source_id.into(),
        message: "a database source requires a connection URL".into(),
    })
}
