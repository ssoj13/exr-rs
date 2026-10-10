//! `exr-rs` — the OpenEXR command line: inspect, convert and view, as subcommands over the
//! verified `exr-core` (see `cli` for the commands). Tools without arguments show help;
//! bare `view`/`v` launches the viewer.

// Every command prints through `cli::stdio` (a closed stdout must not panic); a raw print is an error.
#![deny(clippy::print_stdout, clippy::print_stderr)]

#[macro_use]
mod cli;
#[cfg(test)]
mod guard;

use std::process::ExitCode;

use clap::Parser;
use clap::error::ErrorKind;

/// Process entry point: parse, set up logging and threads, run the command.
///
/// Tools without arguments print help to stdout and exit 0; bare `view`/`v` instead
/// launches the viewer. Help is the CLI's default behaviour, not an error.
/// Real argument errors are clap's (stderr, exit 2);
/// runtime errors are printed by [`cli::run`], which also decides the exit code.
fn main() -> ExitCode {
    let cli = match cli::Cli::try_parse() {
        Ok(cli) => cli,
        Err(e) if e.kind() == ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand => {
            // Styled on a terminal, plain text when redirected.
            let help = e.render();
            if std::io::IsTerminal::is_terminal(&std::io::stdout()) {
                out!("{}", help.ansi());
            } else {
                out!("{help}");
            }
            return ExitCode::SUCCESS;
        }
        Err(e) => e.exit(),
    };
    init_logging(cli.verbose);

    // Chunk-level parallel read/write, as the C++ tools do
    // (`setGlobalThreadCount(ThreadPool::estimateThreadCountForFileIO())`); a failure leaves
    // the library serial, logged.
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) as i32;
    if let Err(e) = exr_core::threading::set_global_thread_count(threads) {
        log::warn!("failed to set the global EXR thread count, continuing serially: {e}");
    }

    // `run` consumed the command line, so everything it owned is dropped before the process ends;
    // this is the only `exit`, and it keeps the C++ tools' `-1` exactly (an `ExitCode` is 0..=255).
    std::process::exit(cli::run(cli))
}

/// Initialise `env_logger`: `RUST_LOG` when set, else `info`, `debug` or `trace` for 0, 1, 2+
/// `-v`.
fn init_logging(verbosity: u8) {
    let default = match verbosity {
        0 => "info",
        1 => "debug",
        _ => "trace",
    };
    env_logger::Builder::new()
        .parse_env(env_logger::Env::default().default_filter_or(default))
        .init();
}
