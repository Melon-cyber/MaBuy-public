use std::sync::atomic::{AtomicU64, AtomicU8, Ordering as AtomicOrdering};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum Side {
    #[default]
    Bid = 0,
    Ask = 1,
}

impl Side {
    #[inline(always)]
    pub const fn opposite(self) -> Self {
        match self {
            Self::Bid => Self::Ask,
            Self::Ask => Self::Bid,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct Level {
    pub price_ticks: i64,
    pub quantity: f64,
    pub timestamp_ns: u64,
}

impl Level {
    pub const EMPTY: Self = Self {
        price_ticks: 0,
        quantity: 0.0,
        timestamp_ns: 0,
    };

    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.quantity <= 0.0
    }
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct L2Delta {
    pub side: Side,
    pub price_ticks: i64,
    pub quantity: f64,
    pub timestamp_ns: u64,
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct DepthChange {
    pub side: Side,
    pub price_ticks: i64,
    pub old_quantity: f64,
    pub new_quantity: f64,
    pub applied: bool,
}

impl DepthChange {
    #[inline(always)]
    pub const fn none(side: Side, price_ticks: i64) -> Self {
        Self {
            side,
            price_ticks,
            old_quantity: 0.0,
            new_quantity: 0.0,
            applied: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BookError {
    InvalidPrice,
    InvalidQuantity,
}

pub struct L2OrderBook<const DEPTH: usize, const DELTA_CAPACITY: usize> {
    tick_size_usd: f64,
    bids: [Level; DEPTH],
    asks: [Level; DEPTH],
    bid_len: usize,
    ask_len: usize,
    deltas: [L2Delta; DELTA_CAPACITY],
    delta_head: usize,
    delta_len: usize,
}

impl<const DEPTH: usize, const DELTA_CAPACITY: usize> L2OrderBook<DEPTH, DELTA_CAPACITY> {
    pub fn new(tick_size_usd: f64) -> Self {
        assert!(DEPTH > 0, "order book depth must be non-zero");
        assert!(
            DELTA_CAPACITY > 0,
            "order book delta capacity must be non-zero"
        );
        assert!(
            tick_size_usd.is_finite() && tick_size_usd > 0.0,
            "tick size must be finite and positive"
        );

        Self {
            tick_size_usd,
            bids: [Level::EMPTY; DEPTH],
            asks: [Level::EMPTY; DEPTH],
            bid_len: 0,
            ask_len: 0,
            deltas: [L2Delta::default(); DELTA_CAPACITY],
            delta_head: 0,
            delta_len: 0,
        }
    }

    #[inline(always)]
    pub fn tick_size_usd(&self) -> f64 {
        self.tick_size_usd
    }

    #[inline(always)]
    pub fn price_usd(&self, price_ticks: i64) -> f64 {
        price_ticks as f64 * self.tick_size_usd
    }

    #[inline(always)]
    pub fn bid_count(&self) -> usize {
        self.bid_len
    }

    #[inline(always)]
    pub fn ask_count(&self) -> usize {
        self.ask_len
    }

    #[inline(always)]
    pub fn bids(&self) -> &[Level] {
        &self.bids[..self.bid_len]
    }

    #[inline(always)]
    pub fn asks(&self) -> &[Level] {
        &self.asks[..self.ask_len]
    }

    #[inline(always)]
    pub fn best_bid(&self) -> Option<Level> {
        if self.bid_len == 0 {
            None
        } else {
            Some(self.bids[0])
        }
    }

    #[inline(always)]
    pub fn best_ask(&self) -> Option<Level> {
        if self.ask_len == 0 {
            None
        } else {
            Some(self.asks[0])
        }
    }

    #[inline(always)]
    pub fn mid_price_usd(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => {
                Some((self.price_usd(bid.price_ticks) + self.price_usd(ask.price_ticks)) * 0.5)
            }
            _ => None,
        }
    }

    #[inline(always)]
    pub fn delta_count(&self) -> usize {
        self.delta_len
    }

    #[inline(always)]
    pub fn latest_delta(&self, age: usize) -> Option<L2Delta> {
        if age >= self.delta_len {
            return None;
        }

        let latest = if self.delta_head == 0 {
            DELTA_CAPACITY - 1
        } else {
            self.delta_head - 1
        };
        let index = (latest + DELTA_CAPACITY - age) % DELTA_CAPACITY;
        Some(self.deltas[index])
    }

    #[inline(always)]
    fn record_delta(&mut self, delta: L2Delta) {
        self.deltas[self.delta_head] = delta;
        self.delta_head += 1;
        if self.delta_head == DELTA_CAPACITY {
            self.delta_head = 0;
        }
        self.delta_len = (self.delta_len + 1).min(DELTA_CAPACITY);
    }

    #[inline(always)]
    fn insertion_index(levels: &[Level; DEPTH], len: usize, side: Side, price_ticks: i64) -> usize {
        let mut index = 0;
        while index < len {
            let existing = levels[index].price_ticks;
            let before = match side {
                Side::Bid => price_ticks > existing,
                Side::Ask => price_ticks < existing,
            };
            if before {
                break;
            }
            index += 1;
        }
        index
    }

    pub fn apply_delta(&mut self, delta: L2Delta) -> Result<DepthChange, BookError> {
        if delta.price_ticks <= 0 {
            return Err(BookError::InvalidPrice);
        }
        if !delta.quantity.is_finite() || delta.quantity < 0.0 {
            return Err(BookError::InvalidQuantity);
        }

        self.record_delta(delta);

        let (levels, len) = match delta.side {
            Side::Bid => (&mut self.bids, &mut self.bid_len),
            Side::Ask => (&mut self.asks, &mut self.ask_len),
        };

        let mut existing_index = None;
        let mut index = 0;
        while index < *len {
            if levels[index].price_ticks == delta.price_ticks {
                existing_index = Some(index);
                break;
            }
            index += 1;
        }

        if let Some(index) = existing_index {
            let old_quantity = levels[index].quantity;

            if delta.quantity == 0.0 {
                let mut shift = index;
                while shift + 1 < *len {
                    levels[shift] = levels[shift + 1];
                    shift += 1;
                }
                *len -= 1;
                levels[*len] = Level::EMPTY;
            } else {
                levels[index].quantity = delta.quantity;
                levels[index].timestamp_ns = delta.timestamp_ns;
            }

            return Ok(DepthChange {
                side: delta.side,
                price_ticks: delta.price_ticks,
                old_quantity,
                new_quantity: delta.quantity,
                applied: true,
            });
        }

        if delta.quantity == 0.0 {
            return Ok(DepthChange::none(delta.side, delta.price_ticks));
        }

        let insert_at = Self::insertion_index(levels, *len, delta.side, delta.price_ticks);
        if *len == DEPTH && insert_at >= DEPTH {
            return Ok(DepthChange::none(delta.side, delta.price_ticks));
        }

        if *len < DEPTH {
            let mut shift = *len;
            while shift > insert_at {
                levels[shift] = levels[shift - 1];
                shift -= 1;
            }
            *len += 1;
        } else {
            let mut shift = DEPTH - 1;
            while shift > insert_at {
                levels[shift] = levels[shift - 1];
                shift -= 1;
            }
        }

        levels[insert_at] = Level {
            price_ticks: delta.price_ticks,
            quantity: delta.quantity,
            timestamp_ns: delta.timestamp_ns,
        };

        Ok(DepthChange {
            side: delta.side,
            price_ticks: delta.price_ticks,
            old_quantity: 0.0,
            new_quantity: delta.quantity,
            applied: true,
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct OfiCalculator {
    cumulative_usd: f64,
    last_increment_usd: f64,
}

impl OfiCalculator {
    #[inline(always)]
    pub fn cumulative_usd(&self) -> f64 {
        self.cumulative_usd
    }

    #[inline(always)]
    pub fn last_increment_usd(&self) -> f64 {
        self.last_increment_usd
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.cumulative_usd = 0.0;
        self.last_increment_usd = 0.0;
    }

    #[inline(always)]
    pub fn observe_change(&mut self, change: DepthChange, tick_size_usd: f64) -> f64 {
        if !change.applied {
            self.last_increment_usd = 0.0;
            return self.cumulative_usd;
        }

        let price_usd = change.price_ticks as f64 * tick_size_usd;
        let quantity_delta = change.new_quantity - change.old_quantity;
        let signed_notional = match change.side {
            Side::Bid => price_usd * quantity_delta,
            Side::Ask => -price_usd * quantity_delta,
        };

        self.last_increment_usd = if signed_notional.is_finite() {
            signed_notional
        } else {
            0.0
        };
        self.cumulative_usd += self.last_increment_usd;
        self.cumulative_usd
    }
}

pub struct HurstKernel<const WINDOW: usize> {
    samples: [f64; WINDOW],
    head: usize,
    len: usize,
}

impl<const WINDOW: usize> HurstKernel<WINDOW> {
    pub fn new() -> Self {
        assert!(WINDOW > 0, "hurst window must be non-zero");
        Self {
            samples: [0.0; WINDOW],
            head: 0,
            len: 0,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.len == WINDOW
    }

    #[inline(always)]
    pub fn push(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }

        self.samples[self.head] = value;
        self.head += 1;
        if self.head == WINDOW {
            self.head = 0;
        }
        self.len = (self.len + 1).min(WINDOW);
    }

    #[inline(always)]
    fn sum_and_squares(&self) -> (f64, f64) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse2") {
                unsafe {
                    return sum_and_squares_sse2(self.samples.as_ptr(), self.len);
                }
            }
        }

        let mut sum = 0.0;
        let mut squares = 0.0;
        let mut index = 0;
        while index < self.len {
            let value = self.samples[index];
            sum += value;
            squares += value * value;
            index += 1;
        }
        (sum, squares)
    }

    pub fn estimate(&self) -> Option<f64> {
        if self.len < 8 {
            return None;
        }

        let n = self.len as f64;
        let (sum, squares) = self.sum_and_squares();
        let mean = sum / n;
        let variance = (squares / n - mean * mean).max(0.0);

        if !variance.is_finite() || variance <= f64::EPSILON {
            return None;
        }

        let standard_deviation = variance.sqrt();
        let start = if self.len == WINDOW { self.head } else { 0 };

        let mut cumulative = 0.0_f64;
        let mut minimum = 0.0_f64;
        let mut maximum = 0.0_f64;
        let mut offset = 0;

        while offset < self.len {
            let index = (start + offset) % WINDOW;
            cumulative += self.samples[index] - mean;
            minimum = minimum.min(cumulative);
            maximum = maximum.max(cumulative);
            offset += 1;
        }

        let range = maximum - minimum;
        if !range.is_finite() || range <= f64::EPSILON {
            return None;
        }

        let rs = range / standard_deviation;
        if !rs.is_finite() || rs <= 0.0 {
            return None;
        }

        let hurst = rs.ln() / n.ln();
        if hurst.is_finite() {
            Some(hurst)
        } else {
            None
        }
    }
}

impl<const WINDOW: usize> Default for HurstKernel<WINDOW> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn sum_and_squares_sse2(values: *const f64, len: usize) -> (f64, f64) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut sum = _mm_setzero_pd();
    let mut squares = _mm_setzero_pd();
    let mut index = 0;

    while index + 2 <= len {
        let value = unsafe { _mm_loadu_pd(values.add(index)) };
        sum = _mm_add_pd(sum, value);
        squares = _mm_add_pd(squares, _mm_mul_pd(value, value));
        index += 2;
    }

    let mut sum_lanes = [0.0_f64; 2];
    let mut square_lanes = [0.0_f64; 2];
    unsafe {
        _mm_storeu_pd(sum_lanes.as_mut_ptr(), sum);
        _mm_storeu_pd(square_lanes.as_mut_ptr(), squares);
    }

    let mut scalar_sum = sum_lanes[0] + sum_lanes[1];
    let mut scalar_squares = square_lanes[0] + square_lanes[1];

    while index < len {
        let value = unsafe { *values.add(index) };
        scalar_sum += value;
        scalar_squares += value * value;
        index += 1;
    }

    (scalar_sum, scalar_squares)
}

pub struct RollingVolatility<const WINDOW: usize> {
    samples: [f64; WINDOW],
    head: usize,
    len: usize,
    sum: f64,
    sum_squares: f64,
}

impl<const WINDOW: usize> RollingVolatility<WINDOW> {
    pub fn new() -> Self {
        assert!(WINDOW > 0, "volatility window must be non-zero");
        Self {
            samples: [0.0; WINDOW],
            head: 0,
            len: 0,
            sum: 0.0,
            sum_squares: 0.0,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn push_return(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }

        if self.len == WINDOW {
            let old = self.samples[self.head];
            self.sum -= old;
            self.sum_squares -= old * old;
        } else {
            self.len += 1;
        }

        self.samples[self.head] = value;
        self.sum += value;
        self.sum_squares += value * value;
        self.head += 1;
        if self.head == WINDOW {
            self.head = 0;
        }
    }

    #[inline(always)]
    pub fn volatility(&self) -> Option<f64> {
        if self.len < 2 {
            return None;
        }

        let n = self.len as f64;
        let mean = self.sum / n;
        let variance = (self.sum_squares / n - mean * mean).max(0.0);

        if variance.is_finite() {
            Some(variance.sqrt())
        } else {
            None
        }
    }
}

impl<const WINDOW: usize> Default for RollingVolatility<WINDOW> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RiskConfig {
    pub hurst_center: f64,
    pub hurst_deadband: f64,
    pub funding_bias_limit: f64,
    pub volatility_trip: f64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            hurst_center: 0.55,
            hurst_deadband: 0.03,
            funding_bias_limit: 0.0005,
            volatility_trip: 0.025,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum HurstRegime {
    Neutral = 0,
    Trending = 1,
    MeanReverting = 2,
}

impl HurstRegime {
    #[inline(always)]
    const fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Trending,
            2 => Self::MeanReverting,
            _ => Self::Neutral,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RiskState {
    Normal = 0,
    LongRestricted = 1,
    ShortRestricted = 2,
    ReduceOnly = 3,
    Halted = 4,
}

impl RiskState {
    #[inline(always)]
    const fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::LongRestricted,
            2 => Self::ShortRestricted,
            3 => Self::ReduceOnly,
            4 => Self::Halted,
            _ => Self::Normal,
        }
    }

    #[inline(always)]
    pub fn allows_open(self, side: Side) -> bool {
        match self {
            Self::Normal => true,
            Self::LongRestricted => side == Side::Ask,
            Self::ShortRestricted => side == Side::Bid,
            Self::ReduceOnly | Self::Halted => false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RiskSnapshot {
    pub state: RiskState,
    pub regime: HurstRegime,
    pub hurst: f64,
    pub funding_rate: f64,
    pub volatility: f64,
    pub generation: u64,
}

pub struct AtomicRiskGate {
    config: RiskConfig,
    state: AtomicU8,
    regime: AtomicU8,
    hurst_bits: AtomicU64,
    funding_bits: AtomicU64,
    volatility_bits: AtomicU64,
    generation: AtomicU64,
}

impl AtomicRiskGate {
    pub fn new(config: RiskConfig) -> Self {
        assert!(config.hurst_center.is_finite());
        assert!(config.hurst_deadband.is_finite() && config.hurst_deadband >= 0.0);
        assert!(config.funding_bias_limit.is_finite() && config.funding_bias_limit >= 0.0);
        assert!(config.volatility_trip.is_finite() && config.volatility_trip > 0.0);

        Self {
            config,
            state: AtomicU8::new(RiskState::ReduceOnly as u8),
            regime: AtomicU8::new(HurstRegime::Neutral as u8),
            hurst_bits: AtomicU64::new(0.5_f64.to_bits()),
            funding_bits: AtomicU64::new(0.0_f64.to_bits()),
            volatility_bits: AtomicU64::new(0.0_f64.to_bits()),
            generation: AtomicU64::new(0),
        }
    }

    #[inline(always)]
    pub fn config(&self) -> RiskConfig {
        self.config
    }

    #[inline(always)]
    pub fn state(&self) -> RiskState {
        RiskState::from_u8(self.state.load(AtomicOrdering::Acquire))
    }

    #[inline(always)]
    pub fn regime(&self) -> HurstRegime {
        HurstRegime::from_u8(self.regime.load(AtomicOrdering::Acquire))
    }

    #[inline(always)]
    pub fn allows_open(&self, side: Side) -> bool {
        self.state().allows_open(side)
    }

    fn transition(&self, desired: RiskState) -> RiskState {
        let mut observed = self.state.load(AtomicOrdering::Acquire);

        loop {
            let current = RiskState::from_u8(observed);
            if current == RiskState::Halted && desired != RiskState::Halted {
                return current;
            }
            if current == desired {
                return current;
            }

            match self.state.compare_exchange_weak(
                observed,
                desired as u8,
                AtomicOrdering::AcqRel,
                AtomicOrdering::Acquire,
            ) {
                Ok(_) => {
                    self.generation.fetch_add(1, AtomicOrdering::Release);
                    return desired;
                }
                Err(next) => observed = next,
            }
        }
    }

    fn update_regime(&self, hurst: f64) -> HurstRegime {
        let current = self.regime();
        let upper = self.config.hurst_center + self.config.hurst_deadband;
        let lower = self.config.hurst_center - self.config.hurst_deadband;

        let next = if hurst >= upper {
            HurstRegime::Trending
        } else if hurst <= lower {
            HurstRegime::MeanReverting
        } else {
            current
        };

        if next != current {
            self.regime.store(next as u8, AtomicOrdering::Release);
            self.generation.fetch_add(1, AtomicOrdering::Release);
        }

        next
    }

    pub fn evaluate(&self, hurst: f64, funding_rate: f64, volatility: f64) -> RiskState {
        self.hurst_bits
            .store(hurst.to_bits(), AtomicOrdering::Release);
        self.funding_bits
            .store(funding_rate.to_bits(), AtomicOrdering::Release);
        self.volatility_bits
            .store(volatility.to_bits(), AtomicOrdering::Release);

        if !volatility.is_finite() || volatility >= self.config.volatility_trip {
            return self.transition(RiskState::Halted);
        }

        if !hurst.is_finite() || !funding_rate.is_finite() {
            return self.transition(RiskState::ReduceOnly);
        }

        let regime = self.update_regime(hurst);
        let desired = match regime {
            HurstRegime::Trending if funding_rate >= self.config.funding_bias_limit => {
                RiskState::LongRestricted
            }
            HurstRegime::Trending if funding_rate <= -self.config.funding_bias_limit => {
                RiskState::ShortRestricted
            }
            HurstRegime::Trending => RiskState::Normal,
            HurstRegime::Neutral | HurstRegime::MeanReverting => RiskState::ReduceOnly,
        };

        self.transition(desired)
    }

    pub fn clear_circuit_breaker(&self) -> bool {
        match self.state.compare_exchange(
            RiskState::Halted as u8,
            RiskState::ReduceOnly as u8,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Acquire,
        ) {
            Ok(_) => {
                self.generation.fetch_add(1, AtomicOrdering::Release);
                true
            }
            Err(_) => false,
        }
    }

    #[inline(always)]
    pub fn snapshot(&self) -> RiskSnapshot {
        RiskSnapshot {
            state: self.state(),
            regime: self.regime(),
            hurst: f64::from_bits(self.hurst_bits.load(AtomicOrdering::Acquire)),
            funding_rate: f64::from_bits(self.funding_bits.load(AtomicOrdering::Acquire)),
            volatility: f64::from_bits(self.volatility_bits.load(AtomicOrdering::Acquire)),
            generation: self.generation.load(AtomicOrdering::Acquire),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct TelemetryEvent {
    pub timestamp_ns: u64,
    pub kind: u32,
    pub flags: u32,
    pub value0: f64,
    pub value1: f64,
    pub value2: f64,
}

struct TelemetrySlot {
    sequence: AtomicU64,
    timestamp_ns: AtomicU64,
    kind_and_flags: AtomicU64,
    value0: AtomicU64,
    value1: AtomicU64,
    value2: AtomicU64,
}

impl TelemetrySlot {
    fn new(sequence: u64) -> Self {
        Self {
            sequence: AtomicU64::new(sequence),
            timestamp_ns: AtomicU64::new(0),
            kind_and_flags: AtomicU64::new(0),
            value0: AtomicU64::new(0),
            value1: AtomicU64::new(0),
            value2: AtomicU64::new(0),
        }
    }

    #[inline(always)]
    fn write(&self, event: TelemetryEvent) {
        self.timestamp_ns
            .store(event.timestamp_ns, AtomicOrdering::Relaxed);
        self.kind_and_flags.store(
            ((event.flags as u64) << 32) | event.kind as u64,
            AtomicOrdering::Relaxed,
        );
        self.value0
            .store(event.value0.to_bits(), AtomicOrdering::Relaxed);
        self.value1
            .store(event.value1.to_bits(), AtomicOrdering::Relaxed);
        self.value2
            .store(event.value2.to_bits(), AtomicOrdering::Relaxed);
    }

    #[inline(always)]
    fn read(&self) -> TelemetryEvent {
        let kind_and_flags = self.kind_and_flags.load(AtomicOrdering::Relaxed);
        TelemetryEvent {
            timestamp_ns: self.timestamp_ns.load(AtomicOrdering::Relaxed),
            kind: kind_and_flags as u32,
            flags: (kind_and_flags >> 32) as u32,
            value0: f64::from_bits(self.value0.load(AtomicOrdering::Relaxed)),
            value1: f64::from_bits(self.value1.load(AtomicOrdering::Relaxed)),
            value2: f64::from_bits(self.value2.load(AtomicOrdering::Relaxed)),
        }
    }
}

pub struct TelemetryRing<const CAPACITY: usize> {
    slots: [TelemetrySlot; CAPACITY],
    enqueue_pos: AtomicU64,
    dequeue_pos: AtomicU64,
}

impl<const CAPACITY: usize> TelemetryRing<CAPACITY> {
    pub fn new() -> Self {
        assert!(CAPACITY > 0, "telemetry capacity must be non-zero");

        Self {
            slots: std::array::from_fn(|index| TelemetrySlot::new(index as u64)),
            enqueue_pos: AtomicU64::new(0),
            dequeue_pos: AtomicU64::new(0),
        }
    }

    pub fn try_push(&self, event: TelemetryEvent) -> Result<(), TelemetryEvent> {
        let mut position = self.enqueue_pos.load(AtomicOrdering::Relaxed);

        loop {
            let slot = &self.slots[(position as usize) % CAPACITY];
            let sequence = slot.sequence.load(AtomicOrdering::Acquire);
            let difference = sequence.wrapping_sub(position) as i64;

            if difference == 0 {
                match self.enqueue_pos.compare_exchange_weak(
                    position,
                    position.wrapping_add(1),
                    AtomicOrdering::Relaxed,
                    AtomicOrdering::Relaxed,
                ) {
                    Ok(_) => {
                        slot.write(event);
                        slot.sequence
                            .store(position.wrapping_add(1), AtomicOrdering::Release);
                        return Ok(());
                    }
                    Err(next) => position = next,
                }
            } else if difference < 0 {
                return Err(event);
            } else {
                position = self.enqueue_pos.load(AtomicOrdering::Relaxed);
            }
        }
    }

    pub fn try_pop(&self) -> Option<TelemetryEvent> {
        let mut position = self.dequeue_pos.load(AtomicOrdering::Relaxed);

        loop {
            let slot = &self.slots[(position as usize) % CAPACITY];
            let sequence = slot.sequence.load(AtomicOrdering::Acquire);
            let expected = position.wrapping_add(1);
            let difference = sequence.wrapping_sub(expected) as i64;

            if difference == 0 {
                match self.dequeue_pos.compare_exchange_weak(
                    position,
                    position.wrapping_add(1),
                    AtomicOrdering::Relaxed,
                    AtomicOrdering::Relaxed,
                ) {
                    Ok(_) => {
                        let event = slot.read();
                        slot.sequence.store(
                            position.wrapping_add(CAPACITY as u64),
                            AtomicOrdering::Release,
                        );
                        return Some(event);
                    }
                    Err(next) => position = next,
                }
            } else if difference < 0 {
                return None;
            } else {
                position = self.dequeue_pos.load(AtomicOrdering::Relaxed);
            }
        }
    }

    #[inline(always)]
    pub fn len_approx(&self) -> usize {
        let pushed = self.enqueue_pos.load(AtomicOrdering::Acquire);
        let popped = self.dequeue_pos.load(AtomicOrdering::Acquire);
        pushed
            .wrapping_sub(popped)
            .min(CAPACITY as u64) as usize
    }

    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        CAPACITY
    }
}

impl<const CAPACITY: usize> Default for TelemetryRing<CAPACITY> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct MicrostructureEngine<
    const DEPTH: usize,
    const DELTA_CAPACITY: usize,
    const HURST_WINDOW: usize,
    const VOLATILITY_WINDOW: usize,
    const TELEMETRY_CAPACITY: usize,
> {
    pub book: L2OrderBook<DEPTH, DELTA_CAPACITY>,
    pub ofi: OfiCalculator,
    pub hurst: HurstKernel<HURST_WINDOW>,
    pub volatility: RollingVolatility<VOLATILITY_WINDOW>,
    pub risk: AtomicRiskGate,
    pub telemetry: TelemetryRing<TELEMETRY_CAPACITY>,
    last_trade_price: Option<f64>,
    funding_rate: f64,
}

impl<
        const DEPTH: usize,
        const DELTA_CAPACITY: usize,
        const HURST_WINDOW: usize,
        const VOLATILITY_WINDOW: usize,
        const TELEMETRY_CAPACITY: usize,
    >
    MicrostructureEngine<
        DEPTH,
        DELTA_CAPACITY,
        HURST_WINDOW,
        VOLATILITY_WINDOW,
        TELEMETRY_CAPACITY,
    >
{
    pub fn new(tick_size_usd: f64, risk_config: RiskConfig) -> Self {
        Self {
            book: L2OrderBook::new(tick_size_usd),
            ofi: OfiCalculator::default(),
            hurst: HurstKernel::new(),
            volatility: RollingVolatility::new(),
            risk: AtomicRiskGate::new(risk_config),
            telemetry: TelemetryRing::new(),
            last_trade_price: None,
            funding_rate: 0.0,
        }
    }

    #[inline(always)]
    pub fn funding_rate(&self) -> f64 {
        self.funding_rate
    }

    #[inline(always)]
    pub fn set_funding_rate(&mut self, funding_rate: f64) {
        if funding_rate.is_finite() {
            self.funding_rate = funding_rate;
        }
    }

    pub fn apply_l2(&mut self, delta: L2Delta) -> Result<DepthChange, BookError> {
        let change = self.book.apply_delta(delta)?;
        let ofi = self.ofi.observe_change(change, self.book.tick_size_usd());

        let _ = self.telemetry.try_push(TelemetryEvent {
            timestamp_ns: delta.timestamp_ns,
            kind: 1,
            flags: delta.side as u32,
            value0: delta.price_ticks as f64,
            value1: delta.quantity,
            value2: ofi,
        });

        Ok(change)
    }

    pub fn observe_trade(&mut self, timestamp_ns: u64, trade_price_usd: f64) -> RiskSnapshot {
        if !trade_price_usd.is_finite() || trade_price_usd <= 0.0 {
            return self.risk.snapshot();
        }

        if let Some(previous) = self.last_trade_price {
            let log_return = (trade_price_usd / previous).ln();
            if log_return.is_finite() {
                self.hurst.push(log_return);
                self.volatility.push_return(log_return);
            }
        }

        self.last_trade_price = Some(trade_price_usd);

        let hurst = self.hurst.estimate().unwrap_or(0.5);
        let volatility = self.volatility.volatility().unwrap_or(0.0);
        let state = self.risk.evaluate(hurst, self.funding_rate, volatility);

        let _ = self.telemetry.try_push(TelemetryEvent {
            timestamp_ns,
            kind: 2,
            flags: state as u32,
            value0: trade_price_usd,
            value1: hurst,
            value2: volatility,
        });

        self.risk.snapshot()
    }
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::prelude::*;

    #[pyclass]
    pub struct PyTelemetryRing {
        inner: TelemetryRing<4096>,
    }

    #[pymethods]
    impl PyTelemetryRing {
        #[new]
        fn new() -> Self {
            Self {
                inner: TelemetryRing::new(),
            }
        }

        fn try_push(
            &self,
            timestamp_ns: u64,
            kind: u32,
            flags: u32,
            value0: f64,
            value1: f64,
            value2: f64,
        ) -> bool {
            self.inner
                .try_push(TelemetryEvent {
                    timestamp_ns,
                    kind,
                    flags,
                    value0,
                    value1,
                    value2,
                })
                .is_ok()
        }

        fn try_pop(&self) -> Option<(u64, u32, u32, f64, f64, f64)> {
            self.inner.try_pop().map(|event| {
                (
                    event.timestamp_ns,
                    event.kind,
                    event.flags,
                    event.value0,
                    event.value1,
                    event.value2,
                )
            })
        }

        fn len_approx(&self) -> usize {
            self.inner.len_approx()
        }

        fn capacity(&self) -> usize {
            self.inner.capacity()
        }
    }

    #[pymodule]
    fn engine_core_py(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
        module.add_class::<PyTelemetryRing>()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_book_sorts_and_updates_without_allocation() {
        let mut book = L2OrderBook::<4, 8>::new(0.01);

        book.apply_delta(L2Delta {
            side: Side::Bid,
            price_ticks: 100,
            quantity: 3.0,
            timestamp_ns: 1,
        })
        .unwrap();
        book.apply_delta(L2Delta {
            side: Side::Bid,
            price_ticks: 102,
            quantity: 1.0,
            timestamp_ns: 2,
        })
        .unwrap();
        book.apply_delta(L2Delta {
            side: Side::Bid,
            price_ticks: 101,
            quantity: 2.0,
            timestamp_ns: 3,
        })
        .unwrap();

        assert_eq!(book.bids()[0].price_ticks, 102);
        assert_eq!(book.bids()[1].price_ticks, 101);
        assert_eq!(book.bids()[2].price_ticks, 100);

        let change = book
            .apply_delta(L2Delta {
                side: Side::Bid,
                price_ticks: 101,
                quantity: 0.0,
                timestamp_ns: 4,
            })
            .unwrap();

        assert_eq!(change.old_quantity, 2.0);
        assert_eq!(book.bid_count(), 2);
        assert_eq!(book.bids()[1].price_ticks, 100);
    }

    #[test]
    fn ofi_is_usd_notional_weighted() {
        let mut ofi = OfiCalculator::default();

        let first = ofi.observe_change(
            DepthChange {
                side: Side::Bid,
                price_ticks: 10_000,
                old_quantity: 0.0,
                new_quantity: 2.0,
                applied: true,
            },
            0.01,
        );

        let second = ofi.observe_change(
            DepthChange {
                side: Side::Ask,
                price_ticks: 10_100,
                old_quantity: 0.0,
                new_quantity: 1.0,
                applied: true,
            },
            0.01,
        );

        assert!((first - 200.0).abs() < 1e-12);
        assert!((second - 99.0).abs() < 1e-12);
    }

    #[test]
    fn hurst_kernel_returns_finite_estimate() {
        let mut kernel = HurstKernel::<64>::new();
        let mut state = 0x1234_5678_u64;

        for _ in 0..64 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let value = ((state >> 11) as f64 / ((u64::MAX >> 11) as f64)) - 0.5;
            kernel.push(value);
        }

        let hurst = kernel.estimate().unwrap();
        assert!(hurst.is_finite());
        assert!(hurst > 0.0);
        assert!(hurst < 2.0);
    }

    #[test]
    fn rolling_volatility_evicts_old_samples() {
        let mut volatility = RollingVolatility::<4>::new();
        volatility.push_return(0.01);
        volatility.push_return(-0.01);
        volatility.push_return(0.02);
        volatility.push_return(-0.02);

        let first = volatility.volatility().unwrap();
        assert!(first > 0.0);

        volatility.push_return(0.0);
        assert_eq!(volatility.len(), 4);
        assert!(volatility.volatility().unwrap().is_finite());
    }

    #[test]
    fn risk_gate_honors_funding_and_latches_circuit_breaker() {
        let gate = AtomicRiskGate::new(RiskConfig {
            hurst_center: 0.5,
            hurst_deadband: 0.05,
            funding_bias_limit: 0.001,
            volatility_trip: 0.02,
        });

        assert_eq!(gate.evaluate(0.70, 0.0, 0.001), RiskState::Normal);
        assert_eq!(
            gate.evaluate(0.70, 0.002, 0.001),
            RiskState::LongRestricted
        );
        assert!(!gate.allows_open(Side::Bid));
        assert!(gate.allows_open(Side::Ask));

        assert_eq!(gate.evaluate(0.70, 0.0, 0.03), RiskState::Halted);
        assert_eq!(gate.evaluate(0.70, 0.0, 0.001), RiskState::Halted);

        assert!(gate.clear_circuit_breaker());
        assert_eq!(gate.evaluate(0.70, 0.0, 0.001), RiskState::Normal);
    }

    #[test]
    fn telemetry_ring_is_bounded_and_fifo() {
        let ring = TelemetryRing::<2>::new();

        assert!(ring
            .try_push(TelemetryEvent {
                timestamp_ns: 1,
                kind: 7,
                flags: 2,
                value0: 1.0,
                value1: 2.0,
                value2: 3.0,
            })
            .is_ok());
        assert!(ring
            .try_push(TelemetryEvent {
                timestamp_ns: 2,
                kind: 8,
                flags: 3,
                value0: 4.0,
                value1: 5.0,
                value2: 6.0,
            })
            .is_ok());
        assert!(ring
            .try_push(TelemetryEvent {
                timestamp_ns: 3,
                kind: 9,
                flags: 4,
                value0: 7.0,
                value1: 8.0,
                value2: 9.0,
            })
            .is_err());

        let first = ring.try_pop().unwrap();
        let second = ring.try_pop().unwrap();

        assert_eq!(first.timestamp_ns, 1);
        assert_eq!(first.kind, 7);
        assert_eq!(second.timestamp_ns, 2);
        assert_eq!(second.kind, 8);
        assert!(ring.try_pop().is_none());
    }

    #[test]
    fn integrated_engine_updates_book_ofi_and_risk() {
        let mut engine =
            MicrostructureEngine::<8, 32, 32, 32, 64>::new(0.01, RiskConfig::default());

        engine
            .apply_l2(L2Delta {
                side: Side::Bid,
                price_ticks: 10_000,
                quantity: 5.0,
                timestamp_ns: 1,
            })
            .unwrap();
        engine
            .apply_l2(L2Delta {
                side: Side::Ask,
                price_ticks: 10_001,
                quantity: 3.0,
                timestamp_ns: 2,
            })
            .unwrap();

        for index in 0..40 {
            let price = 100.0 + index as f64 * 0.01;
            engine.observe_trade(index, price);
        }

        assert!(engine.book.best_bid().is_some());
        assert!(engine.book.best_ask().is_some());
        assert!(engine.ofi.cumulative_usd().is_finite());
        assert!(engine.telemetry.len_approx() > 0);
    }

    #[test]
    fn order_book_insert_update_delete_and_ask_sorting() {
        let mut book = L2OrderBook::<4, 8>::new(0.5);

        assert!(book.best_bid().is_none());
        assert!(book.best_ask().is_none());
        assert!(book.mid_price_usd().is_none());

        // Asks sort ascending (lowest ask is best_ask)
        book.apply_delta(L2Delta {
            side: Side::Ask,
            price_ticks: 205,
            quantity: 4.0,
            timestamp_ns: 1,
        })
        .unwrap();
        book.apply_delta(L2Delta {
            side: Side::Ask,
            price_ticks: 201,
            quantity: 2.0,
            timestamp_ns: 2,
        })
        .unwrap();
        book.apply_delta(L2Delta {
            side: Side::Ask,
            price_ticks: 203,
            quantity: 3.0,
            timestamp_ns: 3,
        })
        .unwrap();

        assert_eq!(book.ask_count(), 3);
        assert_eq!(book.asks()[0].price_ticks, 201);
        assert_eq!(book.asks()[1].price_ticks, 203);
        assert_eq!(book.asks()[2].price_ticks, 205);

        // Update existing ask level
        let upd = book
            .apply_delta(L2Delta {
                side: Side::Ask,
                price_ticks: 201,
                quantity: 5.5,
                timestamp_ns: 4,
            })
            .unwrap();
        assert!(upd.applied);
        assert_eq!(upd.old_quantity, 2.0);
        assert_eq!(upd.new_quantity, 5.5);
        assert_eq!(book.best_ask().unwrap().quantity, 5.5);
        assert_eq!(book.best_ask().unwrap().timestamp_ns, 4);

        // Add best bid and check mid_price_usd
        book.apply_delta(L2Delta {
            side: Side::Bid,
            price_ticks: 199,
            quantity: 1.0,
            timestamp_ns: 5,
        })
        .unwrap();
        // (199 * 0.5 + 201 * 0.5) * 0.5 = 100.0
        assert!((book.mid_price_usd().unwrap() - 100.0).abs() < 1e-12);

        // Delete best ask
        let del = book
            .apply_delta(L2Delta {
                side: Side::Ask,
                price_ticks: 201,
                quantity: 0.0,
                timestamp_ns: 6,
            })
            .unwrap();
        assert!(del.applied);
        assert_eq!(del.old_quantity, 5.5);
        assert_eq!(del.new_quantity, 0.0);
        assert_eq!(book.ask_count(), 2);
        assert_eq!(book.best_ask().unwrap().price_ticks, 203);
    }

    #[test]
    fn order_book_depth_overflow_and_delta_ring_wraparound() {
        let mut book = L2OrderBook::<3, 4>::new(1.0);

        for (i, price) in [100, 102, 101].iter().enumerate() {
            book.apply_delta(L2Delta {
                side: Side::Bid,
                price_ticks: *price,
                quantity: 1.0,
                timestamp_ns: (i + 1) as u64,
            })
            .unwrap();
        }
        assert_eq!(book.bid_count(), 3);

        // Worse than worst bid (100) when full -> ignored (applied = false)
        let ignored = book
            .apply_delta(L2Delta {
                side: Side::Bid,
                price_ticks: 99,
                quantity: 10.0,
                timestamp_ns: 4,
            })
            .unwrap();
        assert!(!ignored.applied);
        assert_eq!(book.bid_count(), 3);
        assert_eq!(book.bids()[2].price_ticks, 100);

        // Better than worst bid -> inserts and evicts worst (100)
        let inserted = book
            .apply_delta(L2Delta {
                side: Side::Bid,
                price_ticks: 103,
                quantity: 2.0,
                timestamp_ns: 5,
            })
            .unwrap();
        assert!(inserted.applied);
        assert_eq!(book.bid_count(), 3);
        assert_eq!(book.bids()[0].price_ticks, 103);
        assert_eq!(book.bids()[1].price_ticks, 102);
        assert_eq!(book.bids()[2].price_ticks, 101);

        // Delta ring buffer capacity is 4, 5 deltas pushed -> wrapped
        assert_eq!(book.delta_count(), 4);
        assert_eq!(book.latest_delta(0).unwrap().timestamp_ns, 5);
        assert_eq!(book.latest_delta(3).unwrap().timestamp_ns, 2);
        assert!(book.latest_delta(4).is_none());
    }

    #[test]
    fn order_book_edge_cases_invalid_and_zero_volume() {
        let mut book = L2OrderBook::<4, 4>::new(0.01);

        assert_eq!(
            book.apply_delta(L2Delta {
                side: Side::Bid,
                price_ticks: 0,
                quantity: 1.0,
                timestamp_ns: 1,
            })
            .unwrap_err(),
            BookError::InvalidPrice
        );
        assert_eq!(
            book.apply_delta(L2Delta {
                side: Side::Ask,
                price_ticks: -5,
                quantity: 1.0,
                timestamp_ns: 1,
            })
            .unwrap_err(),
            BookError::InvalidPrice
        );

        for bad_qty in [-0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                book.apply_delta(L2Delta {
                    side: Side::Bid,
                    price_ticks: 100,
                    quantity: bad_qty,
                    timestamp_ns: 2,
                })
                .unwrap_err(),
                BookError::InvalidQuantity
            );
        }

        // Zero quantity on missing price level is a no-op
        let noop = book
            .apply_delta(L2Delta {
                side: Side::Bid,
                price_ticks: 100,
                quantity: 0.0,
                timestamp_ns: 3,
            })
            .unwrap();
        assert!(!noop.applied);
        assert_eq!(book.bid_count(), 0);
    }

    #[test]
    fn ofi_sign_magnitude_unapplied_and_reset() {
        let mut ofi = OfiCalculator::default();
        let tick = 0.01;

        // Bid add: + 10000 * 0.01 * (3 - 0) = +300
        let c1 = ofi.observe_change(
            DepthChange {
                side: Side::Bid,
                price_ticks: 10_000,
                old_quantity: 0.0,
                new_quantity: 3.0,
                applied: true,
            },
            tick,
        );
        assert!((c1 - 300.0).abs() < 1e-12);
        assert!((ofi.last_increment_usd() - 300.0).abs() < 1e-12);

        // Bid reduce: + 10000 * 0.01 * (1 - 3) = -200 -> cum = 100
        let c2 = ofi.observe_change(
            DepthChange {
                side: Side::Bid,
                price_ticks: 10_000,
                old_quantity: 3.0,
                new_quantity: 1.0,
                applied: true,
            },
            tick,
        );
        assert!((c2 - 100.0).abs() < 1e-12);
        assert!((ofi.last_increment_usd() + 200.0).abs() < 1e-12);

        // Ask add: - 10100 * 0.01 * (2 - 0) = -202 -> cum = -102
        let c3 = ofi.observe_change(
            DepthChange {
                side: Side::Ask,
                price_ticks: 10_100,
                old_quantity: 0.0,
                new_quantity: 2.0,
                applied: true,
            },
            tick,
        );
        assert!((c3 + 102.0).abs() < 1e-12);

        // Ask delete: - 10100 * 0.01 * (0 - 2) = +202 -> cum = 100
        let c4 = ofi.observe_change(
            DepthChange {
                side: Side::Ask,
                price_ticks: 10_100,
                old_quantity: 2.0,
                new_quantity: 0.0,
                applied: true,
            },
            tick,
        );
        assert!((c4 - 100.0).abs() < 1e-12);

        // Unapplied change must keep cumulative_usd and zero last_increment_usd
        let c5 = ofi.observe_change(DepthChange::none(Side::Bid, 9_999), tick);
        assert!((c5 - 100.0).abs() < 1e-12);
        assert_eq!(ofi.last_increment_usd(), 0.0);
        assert!((ofi.cumulative_usd() - 100.0).abs() < 1e-12);

        // Non-finite tick size must not poison cumulative_usd
        let c6 = ofi.observe_change(
            DepthChange {
                side: Side::Bid,
                price_ticks: 10_000,
                old_quantity: 0.0,
                new_quantity: 1.0,
                applied: true,
            },
            f64::INFINITY,
        );
        assert!((c6 - 100.0).abs() < 1e-12);
        assert_eq!(ofi.last_increment_usd(), 0.0);

        ofi.reset();
        assert_eq!(ofi.cumulative_usd(), 0.0);
        assert_eq!(ofi.last_increment_usd(), 0.0);
    }

    #[test]
    fn hurst_distinguishes_trending_random_walk_and_mean_reverting() {
        let mut trend = HurstKernel::<256>::new();
        let mut rw = HurstKernel::<256>::new();
        let mut mr = HurstKernel::<256>::new();

        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next_noise = || -> f64 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 / ((u64::MAX >> 11) as f64)) - 0.5
        };

        let mut r_trend = 0.0;
        let mut r_mr = 0.0;
        for _ in 0..256 {
            let eps = next_noise() * 0.01;
            r_trend = 0.80 * r_trend + eps;
            r_mr = -0.80 * r_mr + eps;
            trend.push(r_trend);
            rw.push(eps);
            mr.push(r_mr);
        }

        let h_trend = trend.estimate().unwrap();
        let h_rw = rw.estimate().unwrap();
        let h_mr = mr.estimate().unwrap();

        assert!(h_trend > 0.65, "expected persistent H > 0.65, got {h_trend}");
        assert!(
            h_rw > 0.40 && h_rw < 0.70,
            "expected random-walk H near 0.5, got {h_rw}"
        );
        assert!(h_mr < 0.40, "expected anti-persistent H < 0.40, got {h_mr}");
        assert!(h_mr < h_rw && h_rw < h_trend);
    }

    #[test]
    fn hurst_edge_cases_and_ring_wraparound() {
        let mut k = HurstKernel::<16>::new();
        assert!(k.estimate().is_none());

        for i in 0..7 {
            k.push(i as f64 * 0.01);
        }
        assert_eq!(k.len(), 7);
        assert!(k.estimate().is_none());

        // Non-finite values are ignored
        k.push(f64::NAN);
        k.push(f64::INFINITY);
        k.push(f64::NEG_INFINITY);
        assert_eq!(k.len(), 7);

        // Zero-variance series returns None
        let mut zero_k = HurstKernel::<16>::new();
        for _ in 0..16 {
            zero_k.push(0.0);
        }
        assert!(zero_k.estimate().is_none());

        // Wraparound consistency: pushing 32 samples into WINDOW=16 matches fresh WINDOW=16 with last 16
        let mut wrapped = HurstKernel::<16>::new();
        let mut fresh = HurstKernel::<16>::new();
        let series: Vec<f64> = (0..32).map(|i| ((i as f64) * 0.37).sin() * 0.01).collect();
        for &v in &series {
            wrapped.push(v);
        }
        for &v in &series[16..] {
            fresh.push(v);
        }
        assert!(wrapped.is_full());
        let h_wrapped = wrapped.estimate().unwrap();
        let h_fresh = fresh.estimate().unwrap();
        assert!((h_wrapped - h_fresh).abs() < 1e-12);
    }

    #[test]
    fn rolling_volatility_matches_hand_computed_and_edge_cases() {
        let mut vol = RollingVolatility::<4>::new();
        assert!(vol.volatility().is_none());

        vol.push_return(1.0);
        assert!(vol.volatility().is_none());

        // Non-finite returns are ignored
        vol.push_return(f64::NAN);
        vol.push_return(f64::INFINITY);
        vol.push_return(f64::NEG_INFINITY);
        assert_eq!(vol.len(), 1);

        vol.push_return(2.0);
        vol.push_return(3.0);
        vol.push_return(4.0);
        // [1, 2, 3, 4]: mean = 2.5, var = ((1+4+9+16)/4 - 6.25) = 1.25
        let expected_std = 1.25_f64.sqrt();
        assert!((vol.volatility().unwrap() - expected_std).abs() < 1e-12);

        // Evict 1.0, push 5.0 -> [2, 3, 4, 5], var = 1.25
        vol.push_return(5.0);
        assert!((vol.volatility().unwrap() - expected_std).abs() < 1e-12);

        // Zero volatility
        let mut zero_vol = RollingVolatility::<4>::new();
        zero_vol.push_return(0.0);
        zero_vol.push_return(0.0);
        zero_vol.push_return(0.0);
        assert_eq!(zero_vol.volatility(), Some(0.0));
    }

    #[test]
    fn risk_gate_state_transitions_and_allows_open() {
        let gate = AtomicRiskGate::new(RiskConfig::default());
        assert_eq!(gate.state(), RiskState::ReduceOnly);
        assert_eq!(gate.regime(), HurstRegime::Neutral);

        // Deadband [0.52, 0.58] stays Neutral -> ReduceOnly
        assert_eq!(gate.evaluate(0.55, 0.0, 0.005), RiskState::ReduceOnly);
        assert_eq!(gate.regime(), HurstRegime::Neutral);

        // Enter Trending -> Normal
        assert_eq!(gate.evaluate(0.59, 0.0, 0.005), RiskState::Normal);
        assert_eq!(gate.regime(), HurstRegime::Trending);
        assert!(gate.allows_open(Side::Bid));
        assert!(gate.allows_open(Side::Ask));

        // Positive funding bias >= 0.0005 -> LongRestricted (only Ask allowed)
        assert_eq!(gate.evaluate(0.60, 0.0005, 0.005), RiskState::LongRestricted);
        assert!(!gate.allows_open(Side::Bid));
        assert!(gate.allows_open(Side::Ask));

        // Negative funding bias <= -0.0005 -> ShortRestricted (only Bid allowed)
        assert_eq!(gate.evaluate(0.60, -0.0005, 0.005), RiskState::ShortRestricted);
        assert!(gate.allows_open(Side::Bid));
        assert!(!gate.allows_open(Side::Ask));

        // Hysteresis: hurst inside deadband keeps Trending
        assert_eq!(gate.evaluate(0.55, 0.0, 0.005), RiskState::Normal);
        assert_eq!(gate.regime(), HurstRegime::Trending);

        // Drop <= 0.52 switches to MeanReverting -> ReduceOnly
        assert_eq!(gate.evaluate(0.51, 0.0, 0.005), RiskState::ReduceOnly);
        assert_eq!(gate.regime(), HurstRegime::MeanReverting);
        assert!(!gate.allows_open(Side::Bid));
        assert!(!gate.allows_open(Side::Ask));

        // Non-finite hurst or funding_rate -> ReduceOnly
        assert_eq!(gate.evaluate(f64::NAN, 0.0, 0.005), RiskState::ReduceOnly);
        assert_eq!(gate.evaluate(0.70, f64::INFINITY, 0.005), RiskState::ReduceOnly);

        // Non-finite volatility trips Halted
        assert_eq!(gate.evaluate(0.70, 0.0, f64::NAN), RiskState::Halted);
        assert!(!gate.allows_open(Side::Bid));
        assert!(!gate.allows_open(Side::Ask));
        assert!(gate.clear_circuit_breaker());
        assert!(!gate.clear_circuit_breaker());
        assert_eq!(gate.state(), RiskState::ReduceOnly);

        let snap = gate.snapshot();
        assert_eq!(snap.state, RiskState::ReduceOnly);
        assert!(snap.generation > 0);
    }

    #[test]
    fn engine_edge_cases_unapplied_l2_and_invalid_trades() {
        let mut engine =
            MicrostructureEngine::<4, 8, 16, 16, 16>::new(0.01, RiskConfig::default());

        engine
            .apply_l2(L2Delta {
                side: Side::Bid,
                price_ticks: 10_000,
                quantity: 2.0,
                timestamp_ns: 1,
            })
            .unwrap();
        // Deleting missing level is unapplied; telemetry OFI must stay 200.0, not drop to 0.0
        let noop = engine
            .apply_l2(L2Delta {
                side: Side::Ask,
                price_ticks: 10_500,
                quantity: 0.0,
                timestamp_ns: 2,
            })
            .unwrap();
        assert!(!noop.applied);

        let ev1 = engine.telemetry.try_pop().unwrap();
        let ev2 = engine.telemetry.try_pop().unwrap();
        assert!((ev1.value2 - 200.0).abs() < 1e-12);
        assert!((ev2.value2 - 200.0).abs() < 1e-12);

        // Invalid trade prices do not poison Hurst/Volatility
        engine.observe_trade(1, 0.0);
        engine.observe_trade(2, -10.0);
        engine.observe_trade(3, f64::NAN);
        engine.observe_trade(4, f64::INFINITY);
        assert_eq!(engine.hurst.len(), 0);
        assert_eq!(engine.volatility.len(), 0);
    }
}

#[cfg(all(test, feature = "nightly_bench"))]
mod benchmarks {
    extern crate test;

    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_l2_delta_update(b: &mut Bencher) {
        let mut book = L2OrderBook::<32, 1024>::new(0.01);
        let mut timestamp = 0_u64;

        b.iter(|| {
            timestamp = timestamp.wrapping_add(1);
            let price = 10_000 + (timestamp % 32) as i64;
            let _ = book.apply_delta(L2Delta {
                side: Side::Bid,
                price_ticks: price,
                quantity: 1.0 + (timestamp % 7) as f64,
                timestamp_ns: timestamp,
            });
        });
    }

    #[bench]
    fn bench_hurst_estimate(b: &mut Bencher) {
        let mut kernel = HurstKernel::<256>::new();
        for index in 0..256 {
            kernel.push((index as f64 * 0.013).sin());
        }

        b.iter(|| test::black_box(kernel.estimate()));
    }

    #[bench]
    fn bench_telemetry_push_pop(b: &mut Bencher) {
        let ring = TelemetryRing::<1024>::new();
        let event = TelemetryEvent {
            timestamp_ns: 1,
            kind: 1,
            flags: 0,
            value0: 1.0,
            value1: 2.0,
            value2: 3.0,
        };

        b.iter(|| {
            let _ = ring.try_push(event);
            let _ = ring.try_pop();
        });
    }
}
