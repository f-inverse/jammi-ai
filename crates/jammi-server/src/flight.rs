use std::net::SocketAddr;

use datafusion::prelude::SessionContext;
use datafusion_flight_sql_server::service::FlightSqlService;

/// Start an Arrow Flight SQL server backed by a DataFusion SessionContext.
pub async fn serve_flight(
    ctx: &SessionContext,
    addr: SocketAddr,
) -> Result<(), Box<dyn std::error::Error>> {
    let service = FlightSqlService::new(ctx.state());
    tracing::info!("Flight SQL server listening on {addr}");
    service.serve(addr.to_string()).await
}
