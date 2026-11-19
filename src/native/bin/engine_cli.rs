use std::hint::black_box;
use std::io::{self, Read};
use std::time::Instant;

use mabuy_engine_core::{HurstKernel, L2Delta, L2OrderBook, OfiCalculator, RollingVolatility, Side};

const MAX_HURST_WINDOW: usize = 16_384;

fn parse_usize(token: Option<&str>, label: &str) -> Result<usize, String> {
    let raw = token.ok_or_else(|| format!("missing {label}"))?;
    raw.parse::<usize>()
        .map_err(|e| format!("invalid {label} '{raw}': {e}"))
}

fn parse_f64(token: Option<&str>, label: &str) -> Result<f64, String> {
    let raw = token.ok_or_else(|| format!("missing {label}"))?;
    raw.parse::<f64>()
        .map_err(|e| format!("invalid {label} '{raw}': {e}"))
}

fn parse_i64(token: Option<&str>, label: &str) -> Result<i64, String> {
    let raw = token.ok_or_else(|| format!("missing {label}"))?;
    raw.parse::<i64>()
        .map_err(|e| format!("invalid {label} '{raw}': {e}"))
}

fn parse_u64(token: Option<&str>, label: &str) -> Result<u64, String> {
    let raw = token.ok_or_else(|| format!("missing {label}"))?;
    raw.parse::<u64>()
        .map_err(|e| format!("invalid {label} '{raw}': {e}"))
}

fn parse_side(token: Option<&str>) -> Result<Side, String> {
    match token.ok_or_else(|| "missing side".to_string())? {
        "B" | "b" | "bid" | "BID" | "0" => Ok(Side::Bid),
        "A" | "a" | "ask" | "ASK" | "1" => Ok(Side::Ask),
        other => Err(format!("invalid side '{other}' (expected B or A)")),
    }
}

fn parse_f64_vec<'a, I: Iterator<Item = &'a str>>(
    iter: &mut I,
    count: usize,
    label: &str,
) -> Result<Vec<f64>, String> {
    let mut out = Vec::with_capacity(count);
    for idx in 0..count {
        out.push(parse_f64(iter.next(), &format!("{label}[{idx}]"))?);
    }
    Ok(out)
}

fn parse_deltas<'a, I: Iterator<Item = &'a str>>(
    iter: &mut I,
    count: usize,
) -> Result<Vec<L2Delta>, String> {
    let mut deltas = Vec::with_capacity(count);
    for idx in 0..count {
        let side = parse_side(iter.next())?;
        let price_ticks = parse_i64(iter.next(), &format!("price_ticks[{idx}]"))?;
        let quantity = parse_f64(iter.next(), &format!("quantity[{idx}]"))?;
        let timestamp_ns = parse_u64(iter.next(), &format!("timestamp_ns[{idx}]"))?;
        deltas.push(L2Delta {
            side,
            price_ticks,
            quantity,
            timestamp_ns,
        });
    }
    Ok(deltas)
}

fn run_vol_for_window<const W: usize>(samples: &[f64]) -> Option<f64> {
    let mut vol = RollingVolatility::<W>::new();
    for &x in samples {
        vol.push_return(x);
    }
    vol.volatility()
}

fn dispatch_vol(window: usize, samples: &[f64]) -> Result<Option<f64>, String> {
    macro_rules! match_windows {
        ($($w:expr),+ $(,)?) => {
            match window {
                $($w => Ok(run_vol_for_window::<$w>(samples)),)+
                _ => Err(format!("unsupported vol window {window}")),
            }
        };
    }

    match_windows!(
        2, 3, 4, 5, 8, 10, 12, 16, 20, 24, 30, 32, 36, 50, 64, 100, 128, 200, 256, 500, 512, 1000,
        1024
    )
}

fn format_opt(val: Option<f64>) -> String {
    match val {
        Some(v) => format!("{v:.17}"),
        None => "none".to_string(),
    }
}

fn handle_line(line: &str) -> Result<String, String> {
    let mut tokens = line.split_whitespace();
    let cmd = match tokens.next() {
        Some(c) if !c.starts_with('#') => c,
        _ => return Ok(String::new()),
    };

    match cmd {
        "hurst" => {
            let n = parse_usize(tokens.next(), "n")?;
            if n > MAX_HURST_WINDOW {
                return Err(format!("n={n} exceeds MAX_HURST_WINDOW={MAX_HURST_WINDOW}"));
            }
            let values = parse_f64_vec(&mut tokens, n, "sample")?;
            let mut kernel = HurstKernel::<MAX_HURST_WINDOW>::new();
            for v in values {
                kernel.push(v);
            }
            Ok(format!("hurst {}", format_opt(kernel.estimate())))
        }
        "hurst_prices" => {
            let n = parse_usize(tokens.next(), "n")?;
            if n > MAX_HURST_WINDOW + 1 {
                return Err(format!(
                    "n={n} exceeds MAX_HURST_WINDOW+1={}",
                    MAX_HURST_WINDOW + 1
                ));
            }
            let prices = parse_f64_vec(&mut tokens, n, "price")?;
            let mut kernel = HurstKernel::<MAX_HURST_WINDOW>::new();
            let mut prev: Option<f64> = None;
            for p in prices {
                if !p.is_finite() || p <= 0.0 {
                    continue;
                }
                if let Some(last) = prev {
                    let lr = (p / last).ln();
                    if lr.is_finite() {
                        kernel.push(lr);
                    }
                }
                prev = Some(p);
            }
            Ok(format!("hurst_prices {}", format_opt(kernel.estimate())))
        }
        "vol" => {
            let window = parse_usize(tokens.next(), "window")?;
            let n = parse_usize(tokens.next(), "n")?;
            let samples = parse_f64_vec(&mut tokens, n, "return")?;
            let res = dispatch_vol(window, &samples)?;
            Ok(format!("vol {}", format_opt(res)))
        }
        "ofi" => {
            let tick_size = parse_f64(tokens.next(), "tick_size")?;
            let count = parse_usize(tokens.next(), "count")?;
            let deltas = parse_deltas(&mut tokens, count)?;
            let mut book = L2OrderBook::<32, 1024>::new(tick_size);
            let mut ofi = OfiCalculator::default();
            for delta in deltas {
                let change = book
                    .apply_delta(delta)
                    .map_err(|e| format!("book error: {e:?}"))?;
                ofi.observe_change(change, book.tick_size_usd());
            }
            let mut out = format!(
                "ofi {:.17} {:.17} bids {}",
                ofi.cumulative_usd(),
                ofi.last_increment_usd(),
                book.bid_count()
            );
            for lvl in book.bids() {
                out.push_str(&format!(
                    " {:.17}:{:.17}",
                    book.price_usd(lvl.price_ticks),
                    lvl.quantity
                ));
            }
            out.push_str(&format!(" asks {}", book.ask_count()));
            for lvl in book.asks() {
                out.push_str(&format!(
                    " {:.17}:{:.17}",
                    book.price_usd(lvl.price_ticks),
                    lvl.quantity
                ));
            }
            Ok(out)
        }
        "bench_hurst" => {
            let reps = parse_usize(tokens.next(), "reps")?;
            let n = parse_usize(tokens.next(), "n")?;
            if n > MAX_HURST_WINDOW {
                return Err(format!("n={n} exceeds MAX_HURST_WINDOW={MAX_HURST_WINDOW}"));
            }
            let values = parse_f64_vec(&mut tokens, n, "sample")?;

            // Measure full push + estimate cycle
            let mut last = None;
            let t0 = Instant::now();
            for _ in 0..reps {
                let mut kernel = HurstKernel::<MAX_HURST_WINDOW>::new();
                for &v in &values {
                    kernel.push(black_box(v));
                }
                last = black_box(kernel.estimate());
            }
            let full_sec = t0.elapsed().as_secs_f64();

            // Measure estimate() alone on populated kernel
            let mut kernel = HurstKernel::<MAX_HURST_WINDOW>::new();
            for &v in &values {
                kernel.push(v);
            }
            let t1 = Instant::now();
            for _ in 0..reps {
                last = black_box(black_box(&kernel).estimate());
            }
            let eval_sec = t1.elapsed().as_secs_f64();

            Ok(format!(
                "bench_hurst {} {} {:.9} {:.9} {}",
                reps,
                n,
                full_sec,
                eval_sec,
                format_opt(last)
            ))
        }
        "bench_ofi" => {
            let reps = parse_usize(tokens.next(), "reps")?;
            let tick_size = parse_f64(tokens.next(), "tick_size")?;
            let count = parse_usize(tokens.next(), "count")?;
            let deltas = parse_deltas(&mut tokens, count)?;

            let mut last_cum = 0.0_f64;
            let t0 = Instant::now();
            for _ in 0..reps {
                let mut book = L2OrderBook::<32, 1024>::new(tick_size);
                let mut ofi = OfiCalculator::default();
                for &delta in &deltas {
                    if let Ok(change) = book.apply_delta(black_box(delta)) {
                        last_cum = black_box(ofi.observe_change(change, book.tick_size_usd()));
                    }
                }
            }
            let elapsed_sec = t0.elapsed().as_secs_f64();

            Ok(format!(
                "bench_ofi {} {} {:.9} {:.17}",
                reps, count, elapsed_sec, last_cum
            ))
        }
        other => Err(format!("unknown command '{other}'")),
    }
}

fn main() {
    let mut input = String::new();
    if let Err(err) = io::stdin().read_to_string(&mut input) {
        eprintln!("failed to read stdin: {err}");
        std::process::exit(1);
    }

    for (line_no, line) in input.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        match handle_line(trimmed) {
            Ok(out) if !out.is_empty() => println!("{out}"),
            Ok(_) => {}
            Err(err) => {
                eprintln!("line {}: {err}", line_no + 1);
                std::process::exit(2);
            }
        }
    }
}
