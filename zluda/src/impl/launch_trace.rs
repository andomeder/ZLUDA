// A breakdown of what a single launch costs on the host, for
// ZLUDA_TRACE_LAUNCH_MS=1 (one line per launch) or =2 (a summary every
// SUMMARY_EVERY launches).
//
// The per-launch form is the right one for a harness. In an application it is
// not: an unbuffered stderr write costs about 24 us, so printing one line per
// launch changes the very queue behaviour being measured. The summary form keeps
// the instrument out of the measurement.
//
// The page path is host bound per command (about 219000 command activities per
// call at roughly 4.7 us each), so a single number per launch cannot say where
// to cut. This splits one launch into the phases it actually goes through, all
// of them host time on the calling thread:
//
//   gather   collecting an oversized argument block (outside the lock)
//   lock     waiting for the mutex and for queue depth
//   event    taking a signal event from the stream's slot pool
//   reuse    recycling events that have retired
//   set_arg  one entry point call per kernel argument
//   shared   the dynamic shared memory argument
//   group    the group size call
//   append   the Level Zero append itself
//   book     pushing the event onto the stream's pending list
//
// A phase that is not entered reads zero, so a kernel that takes the argument
// buffer path shows gather and one set_arg rather than one per parameter. The
// counters are cells so that the closure that fills them can hold a shared
// reference: the launch path has one borrow and the phases accumulate into it.
use std::cell::Cell;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Launches between two summary lines in the ZLUDA_TRACE_LAUNCH_MS=2 form.
const SUMMARY_EVERY: u64 = 1000;

#[derive(Default, Clone, Copy)]
struct Sums {
    launches: u64,
    total: f64,
    gather: f64,
    lock: f64,
    event: f64,
    reuse: f64,
    set_arg: f64,
    shared: f64,
    group: f64,
    append: f64,
    book: f64,
}

/// ZLUDA_TRACE_LAUNCH_MS=2 asks for the summary form.
fn summary() -> bool {
    static SUMMARY: OnceLock<bool> = OnceLock::new();
    *SUMMARY.get_or_init(|| {
        std::env::var("ZLUDA_TRACE_LAUNCH_MS")
            .map(|value| value == "2")
            .unwrap_or(false)
    })
}

fn sums() -> &'static Mutex<Sums> {
    static SUMS: OnceLock<Mutex<Sums>> = OnceLock::new();
    SUMS.get_or_init(Default::default)
}

#[derive(Default)]
pub struct Trace {
    pub enabled: bool,
    pub gather: Cell<f64>,
    pub lock: Cell<f64>,
    pub event: Cell<f64>,
    pub reuse: Cell<f64>,
    pub set_arg: Cell<f64>,
    pub shared: Cell<f64>,
    pub group: Cell<f64>,
    pub append: Cell<f64>,
    pub book: Cell<f64>,
}

impl Trace {
    pub fn new(enabled: bool) -> Self {
        Trace {
            enabled,
            ..Default::default()
        }
    }

    /// Start a phase; None when tracing is off, so a caller pays a branch.
    pub fn start(&self) -> Option<Instant> {
        if self.enabled {
            Some(Instant::now())
        } else {
            None
        }
    }

    pub fn add(&self, mark: Option<Instant>, into: fn(&Trace) -> &Cell<f64>) {
        if let Some(mark) = mark {
            let cell = into(self);
            cell.set(cell.get() + mark.elapsed().as_secs_f64() * 1e3);
        }
    }

    pub fn total(&self) -> f64 {
        self.gather.get()
            + self.lock.get()
            + self.event.get()
            + self.reuse.get()
            + self.set_arg.get()
            + self.shared.get()
            + self.group.get()
            + self.append.get()
            + self.book.get()
    }

    pub fn print(&self, name: &str, total: f64) {
        if !self.enabled {
            return;
        }
        if summary() {
            let mut sums = sums().lock().unwrap();
            sums.launches += 1;
            sums.total += total;
            sums.gather += self.gather.get();
            sums.lock += self.lock.get();
            sums.event += self.event.get();
            sums.reuse += self.reuse.get();
            sums.set_arg += self.set_arg.get();
            sums.shared += self.shared.get();
            sums.group += self.group.get();
            sums.append += self.append.get();
            sums.book += self.book.get();
            if sums.launches % SUMMARY_EVERY != 0 {
                return;
            }
            let n = sums.launches as f64;
            eprintln!(
                "launch summary: {} launches, mean total {:.3} us, other {:.3}, gather {:.3}, \
                 lock {:.3}, event {:.3}, reuse {:.3}, setarg {:.3}, shared {:.3}, group {:.3}, \
                 append {:.3}, book {:.3}",
                sums.launches,
                sums.total / n * 1e3,
                (sums.total - sums.gather - sums.lock - sums.event - sums.reuse - sums.set_arg
                    - sums.shared - sums.group - sums.append - sums.book) / n * 1e3,
                sums.gather / n * 1e3,
                sums.lock / n * 1e3,
                sums.event / n * 1e3,
                sums.reuse / n * 1e3,
                sums.set_arg / n * 1e3,
                sums.shared / n * 1e3,
                sums.group / n * 1e3,
                sums.append / n * 1e3,
                sums.book / n * 1e3
            );
            return;
        }
        eprintln!(
            "launch ms: total {:.4} other {:.4} gather {:.4} lock {:.4} event {:.4} \
             reuse {:.4} setarg {:.4} shared {:.4} group {:.4} append {:.4} \
             book {:.4} {}",
            total,
            total - self.total(),
            self.gather.get(),
            self.lock.get(),
            self.event.get(),
            self.reuse.get(),
            self.set_arg.get(),
            self.shared.get(),
            self.group.get(),
            self.append.get(),
            self.book.get(),
            name
        );
    }
}
