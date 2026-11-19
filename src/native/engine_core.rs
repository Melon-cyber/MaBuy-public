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

pub mod ffi {
    use std::ffi::c_void;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::slice;

    use super::{L2Delta, MicrostructureEngine, RiskConfig, Side};

    pub type FfiEngine = MicrostructureEngine<32, 1024, 256, 64, 4096>;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct ObiSnapshotC {
        pub obi_score: f64,
        pub microprice: f64,
        pub depth_microprice: f64,
        pub total_bid_vol: f64,
        pub total_ask_vol: f64,
        pub bid_ask_ratio: f64,
        pub bid_wall_price: f64,
        pub ask_wall_price: f64,
        pub has_bid_wall: u8,
        pub has_ask_wall: u8,
        pub has_bid_wall_price: u8,
        pub has_ask_wall_price: u8,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct EngineSnapshotC {
        pub cumulative_ofi_usd: f64,
        pub last_ofi_increment_usd: f64,
        pub mid_price_usd: f64,
        pub has_mid_price: u8,
        pub hurst: f64,
        pub has_hurst: u8,
        pub volatility: f64,
        pub has_volatility: u8,
        pub risk_state: u8,
        pub hurst_regime: u8,
        pub bid_count: usize,
        pub ask_count: usize,
        pub generation: u64,
    }

    fn collect_positive_log_returns(prices: &[f64]) -> (usize, Vec<f64>) {
        let mut log_returns = Vec::with_capacity(prices.len());
        let mut prev_ln: Option<f64> = None;
        let mut valid_count: usize = 0;

        for &p in prices {
            if p.is_finite() && p > 0.0 {
                valid_count += 1;
                let cur_ln = p.ln();
                if let Some(prev) = prev_ln {
                    log_returns.push(cur_ln - prev);
                }
                prev_ln = Some(cur_ln);
            }
        }

        (valid_count, log_returns)
    }

    fn calc_rs_ddof1_2scale(returns: &[f64]) -> f64 {
        if returns.len() < 4 {
            return 1.0;
        }
        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;
        let var = returns
            .iter()
            .map(|&r| {
                let d = r - mean;
                d * d
            })
            .sum::<f64>()
            / ((returns.len() - 1) as f64);
        let s = var.sqrt();
        if !s.is_finite() || s <= 1e-8 {
            return 1.0;
        }

        let mut cum = 0.0_f64;
        let mut min_cum = f64::INFINITY;
        let mut max_cum = f64::NEG_INFINITY;
        for &r in returns {
            cum += r - mean;
            if cum < min_cum {
                min_cum = cum;
            }
            if cum > max_cum {
                max_cum = cum;
            }
        }

        let rs = (max_cum - min_cum) / s;
        if rs.is_finite() {
            rs.max(1e-4)
        } else {
            1.0
        }
    }

    fn estimate_hurst_single_scale_slice(samples: &[f64]) -> Option<f64> {
        if samples.len() < 8 {
            return None;
        }

        let n = samples.len() as f64;
        let (sum, squares) = {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if std::is_x86_feature_detected!("sse2") {
                    unsafe { super::sum_and_squares_sse2(samples.as_ptr(), samples.len()) }
                } else {
                    let mut s = 0.0;
                    let mut sq = 0.0;
                    for &v in samples {
                        s += v;
                        sq += v * v;
                    }
                    (s, sq)
                }
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            {
                let mut s = 0.0;
                let mut sq = 0.0;
                for &v in samples {
                    s += v;
                    sq += v * v;
                }
                (s, sq)
            }
        };

        let mean = sum / n;
        let variance = (squares / n - mean * mean).max(0.0);
        if !variance.is_finite() || variance <= f64::EPSILON {
            return None;
        }

        let standard_deviation = variance.sqrt();
        let mut cumulative = 0.0_f64;
        let mut minimum = 0.0_f64;
        let mut maximum = 0.0_f64;
        for &value in samples {
            cumulative += value - mean;
            minimum = minimum.min(cumulative);
            maximum = maximum.max(cumulative);
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

    #[no_mangle]
    pub extern "C" fn mabuy_hurst_rs_multiscale(
        prices_ptr: *const f64,
        len: usize,
        min_chunk: usize,
    ) -> f64 {
        catch_unwind(AssertUnwindSafe(|| {
            if prices_ptr.is_null() || len < 20 {
                return 0.50;
            }
            let prices = unsafe { slice::from_raw_parts(prices_ptr, len) };
            let (valid_count, log_returns) = collect_positive_log_returns(prices);
            if valid_count < 20 {
                return 0.50;
            }

            let n_total = log_returns.len();
            if n_total < 16 {
                return 0.50;
            }

            let mc = if min_chunk == 0 { 8 } else { min_chunk };
            let max_chunk = n_total / 2;
            let mid_chunk = ((mc as f64) * 1.5) as usize;

            let mut chunk_sizes = Vec::with_capacity(3);
            for c in [mc, mid_chunk, max_chunk] {
                if c >= mc && c <= max_chunk {
                    chunk_sizes.push(c);
                }
            }
            chunk_sizes.sort_unstable();
            chunk_sizes.dedup();
            if chunk_sizes.len() < 2 {
                return 0.50;
            }

            let mut log_chunks = Vec::with_capacity(chunk_sizes.len());
            let mut log_rs_means = Vec::with_capacity(chunk_sizes.len());

            for &n in &chunk_sizes {
                let num_subs = n_total / n;
                let mut rs_sum = 0.0_f64;
                let mut rs_count = 0_usize;

                for i in 0..num_subs {
                    let sub = &log_returns[i * n..(i + 1) * n];
                    let nf = n as f64;
                    let mean = sub.iter().sum::<f64>() / nf;
                    let s = if n > 1 {
                        let var = sub
                            .iter()
                            .map(|&x| {
                                let d = x - mean;
                                d * d
                            })
                            .sum::<f64>()
                            / ((n - 1) as f64);
                        var.sqrt()
                    } else {
                        0.0
                    };

                    if s.is_finite() && s > 1e-12 {
                        let mut cum = 0.0_f64;
                        let mut min_cum = f64::INFINITY;
                        let mut max_cum = f64::NEG_INFINITY;
                        for &x in sub {
                            cum += x - mean;
                            if cum < min_cum {
                                min_cum = cum;
                            }
                            if cum > max_cum {
                                max_cum = cum;
                            }
                        }
                        let r = max_cum - min_cum;
                        if r.is_finite() && r > 1e-12 {
                            rs_sum += r / s;
                            rs_count += 1;
                        }
                    }
                }

                if rs_count > 0 {
                    let mean_rs = rs_sum / (rs_count as f64);
                    if mean_rs.is_finite() && mean_rs > 0.0 {
                        log_chunks.push((n as f64).ln());
                        log_rs_means.push(mean_rs.ln());
                    }
                }
            }

            if log_chunks.len() >= 2 {
                let k = log_chunks.len() as f64;
                let mean_x = log_chunks.iter().sum::<f64>() / k;
                let mean_y = log_rs_means.iter().sum::<f64>() / k;
                let mut num = 0.0_f64;
                let mut den = 0.0_f64;
                for (&x, &y) in log_chunks.iter().zip(log_rs_means.iter()) {
                    let dx = x - mean_x;
                    num += dx * (y - mean_y);
                    den += dx * dx;
                }
                if den.is_finite() && den > 1e-15 {
                    let slope = num / den;
                    if slope.is_finite() {
                        return slope.clamp(0.05, 0.95);
                    }
                }
            }

            0.50
        }))
        .unwrap_or(0.50)
    }

    #[no_mangle]
    pub extern "C" fn mabuy_hurst_2scale(
        prices_ptr: *const f64,
        len: usize,
        min_len: usize,
    ) -> f64 {
        catch_unwind(AssertUnwindSafe(|| {
            if prices_ptr.is_null() || len == 0 || len < min_len {
                return 0.50;
            }
            let prices = unsafe { slice::from_raw_parts(prices_ptr, len) };
            let (valid_count, returns) = collect_positive_log_returns(prices);
            if valid_count < min_len || returns.len() < 12 || returns.iter().all(|&r| r == 0.0) {
                return 0.50;
            }

            let rs_full = calc_rs_ddof1_2scale(&returns);
            let half = returns.len() / 2;
            let rs_half1 = calc_rs_ddof1_2scale(&returns[..half]);
            let rs_half2 = calc_rs_ddof1_2scale(&returns[half..]);
            let rs_half_mean = ((rs_half1 + rs_half2) * 0.5).max(1e-4);

            let ratio = (rs_full / rs_half_mean).max(1e-4);
            let hurst = ratio.ln() / 2.0_f64.ln();
            if hurst.is_finite() {
                hurst.clamp(0.05, 0.95)
            } else {
                0.50
            }
        }))
        .unwrap_or(0.50)
    }

    #[no_mangle]
    pub extern "C" fn mabuy_hurst_single_scale(
        returns_ptr: *const f64,
        len: usize,
        out_val: *mut f64,
    ) -> i32 {
        catch_unwind(AssertUnwindSafe(|| {
            if returns_ptr.is_null() || out_val.is_null() {
                return -1;
            }
            if len < 8 {
                unsafe {
                    *out_val = f64::NAN;
                }
                return 0;
            }
            let raw = unsafe { slice::from_raw_parts(returns_ptr, len) };
            let est = if raw.iter().all(|v| v.is_finite()) {
                estimate_hurst_single_scale_slice(raw)
            } else {
                let finite: Vec<f64> = raw.iter().copied().filter(|v| v.is_finite()).collect();
                estimate_hurst_single_scale_slice(&finite)
            };

            match est {
                Some(v) => {
                    unsafe {
                        *out_val = v;
                    }
                    1
                }
                None => {
                    unsafe {
                        *out_val = f64::NAN;
                    }
                    0
                }
            }
        }))
        .unwrap_or(-1)
    }

    #[no_mangle]
    pub extern "C" fn mabuy_rolling_volatility(
        returns_ptr: *const f64,
        len: usize,
        window: usize,
        out_val: *mut f64,
    ) -> i32 {
        catch_unwind(AssertUnwindSafe(|| {
            if returns_ptr.is_null() || out_val.is_null() {
                return -1;
            }
            if window < 2 || len < 2 {
                unsafe {
                    *out_val = f64::NAN;
                }
                return 0;
            }
            let raw = unsafe { slice::from_raw_parts(returns_ptr, len) };

            let mut count = 0_usize;
            let mut start_idx = len;
            for idx in (0..len).rev() {
                if raw[idx].is_finite() {
                    count += 1;
                    start_idx = idx;
                    if count == window {
                        break;
                    }
                }
            }

            if count < 2 {
                unsafe {
                    *out_val = f64::NAN;
                }
                return 0;
            }

            let nf = count as f64;
            let mut sum = 0.0_f64;
            for &v in &raw[start_idx..] {
                if v.is_finite() {
                    sum += v;
                }
            }
            let mean = sum / nf;
            let mut sq_sum = 0.0_f64;
            for &v in &raw[start_idx..] {
                if v.is_finite() {
                    let d = v - mean;
                    sq_sum += d * d;
                }
            }
            let vol = (sq_sum / nf).sqrt();
            if vol.is_finite() {
                unsafe {
                    *out_val = vol;
                }
                1
            } else {
                unsafe {
                    *out_val = f64::NAN;
                }
                0
            }
        }))
        .unwrap_or(-1)
    }

    #[no_mangle]
    pub extern "C" fn mabuy_obi_snapshot(
        bid_prices: *const f64,
        bid_qtys: *const f64,
        n_bids: usize,
        ask_prices: *const f64,
        ask_qtys: *const f64,
        n_asks: usize,
        min_wall_usd: f64,
        out: *mut ObiSnapshotC,
    ) -> i32 {
        catch_unwind(AssertUnwindSafe(|| {
            if out.is_null() {
                return -1;
            }
            if n_bids > 0 && (bid_prices.is_null() || bid_qtys.is_null()) {
                return -1;
            }
            if n_asks > 0 && (ask_prices.is_null() || ask_qtys.is_null()) {
                return -1;
            }

            if n_bids == 0 || n_asks == 0 {
                unsafe {
                    *out = ObiSnapshotC {
                        obi_score: 0.0,
                        microprice: 0.0,
                        depth_microprice: 0.0,
                        total_bid_vol: 0.0,
                        total_ask_vol: 0.0,
                        bid_ask_ratio: 1.0,
                        bid_wall_price: 0.0,
                        ask_wall_price: 0.0,
                        has_bid_wall: 0,
                        has_ask_wall: 0,
                        has_bid_wall_price: 0,
                        has_ask_wall_price: 0,
                    };
                }
                return 0;
            }

            let nb = n_bids.min(20);
            let na = n_asks.min(20);
            let bp = unsafe { slice::from_raw_parts(bid_prices, nb) };
            let bq = unsafe { slice::from_raw_parts(bid_qtys, nb) };
            let ap = unsafe { slice::from_raw_parts(ask_prices, na) };
            let aq = unsafe { slice::from_raw_parts(ask_qtys, na) };

            let mut total_bid_vol = 0.0_f64;
            for &q in bq {
                total_bid_vol += q;
            }
            let mut total_ask_vol = 0.0_f64;
            for &q in aq {
                total_ask_vol += q;
            }

            let denom = total_bid_vol + total_ask_vol;
            let obi_score = if denom <= 0.0 {
                0.0
            } else {
                (total_bid_vol - total_ask_vol) / denom
            };

            let mut has_bid_wall = 0_u8;
            let mut has_bid_wall_price = 0_u8;
            let mut bid_wall_price = 0.0_f64;
            if nb >= 4 && total_bid_vol > 0.0 {
                for i in 0..nb {
                    let p = bp[i];
                    let q = bq[i];
                    if (q / total_bid_vol) >= 0.35 && (p * q) >= min_wall_usd {
                        has_bid_wall = 1;
                        has_bid_wall_price = 1;
                        bid_wall_price = p;
                        break;
                    }
                }
            }

            let mut has_ask_wall = 0_u8;
            let mut has_ask_wall_price = 0_u8;
            let mut ask_wall_price = 0.0_f64;
            if na >= 4 && total_ask_vol > 0.0 {
                for i in 0..na {
                    let p = ap[i];
                    let q = aq[i];
                    if (q / total_ask_vol) >= 0.35 && (p * q) >= min_wall_usd {
                        has_ask_wall = 1;
                        has_ask_wall_price = 1;
                        ask_wall_price = p;
                        break;
                    }
                }
            }

            let top_denom = bq[0] + aq[0];
            let microprice = if top_denom <= 0.0 {
                (bp[0] + ap[0]) * 0.5
            } else {
                (ap[0] * bq[0] + bp[0] * aq[0]) / top_denom
            };

            let n_levels = nb.min(na).min(5);
            let mut depth_num = 0.0_f64;
            let mut depth_den = 0.0_f64;
            for i in 0..n_levels {
                depth_num += ap[i] * bq[i] + bp[i] * aq[i];
                depth_den += bq[i] + aq[i];
            }
            let depth_microprice = if depth_den <= 0.0 {
                (bp[0] + ap[0]) * 0.5
            } else {
                depth_num / depth_den
            };

            let bid_ask_ratio = total_bid_vol / total_ask_vol.max(0.0001);

            unsafe {
                *out = ObiSnapshotC {
                    obi_score,
                    microprice,
                    depth_microprice,
                    total_bid_vol,
                    total_ask_vol,
                    bid_ask_ratio,
                    bid_wall_price,
                    ask_wall_price,
                    has_bid_wall,
                    has_ask_wall,
                    has_bid_wall_price,
                    has_ask_wall_price,
                };
            }
            0
        }))
        .unwrap_or(-1)
    }

    #[no_mangle]
    pub extern "C" fn mabuy_engine_new(
        tick_size_usd: f64,
        hurst_center: f64,
        hurst_deadband: f64,
        funding_bias_limit: f64,
        volatility_trip: f64,
    ) -> *mut c_void {
        catch_unwind(AssertUnwindSafe(|| {
            if !tick_size_usd.is_finite()
                || tick_size_usd <= 0.0
                || !hurst_center.is_finite()
                || !hurst_deadband.is_finite()
                || hurst_deadband < 0.0
                || !funding_bias_limit.is_finite()
                || funding_bias_limit < 0.0
                || !volatility_trip.is_finite()
                || volatility_trip <= 0.0
            {
                return std::ptr::null_mut();
            }

            let config = RiskConfig {
                hurst_center,
                hurst_deadband,
                funding_bias_limit,
                volatility_trip,
            };
            let engine = Box::new(FfiEngine::new(tick_size_usd, config));
            Box::into_raw(engine) as *mut c_void
        }))
        .unwrap_or(std::ptr::null_mut())
    }

    #[no_mangle]
    pub extern "C" fn mabuy_engine_free(ptr: *mut c_void) {
        if ptr.is_null() {
            return;
        }
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
            drop(Box::from_raw(ptr as *mut FfiEngine));
        }));
    }

    #[no_mangle]
    pub extern "C" fn mabuy_engine_apply_l2_batch(
        ptr: *mut c_void,
        sides: *const u8,
        price_ticks: *const i64,
        quantities: *const f64,
        timestamps_ns: *const u64,
        count: usize,
        out_cum_ofi: *mut f64,
    ) -> i32 {
        catch_unwind(AssertUnwindSafe(|| {
            if ptr.is_null() || count > i32::MAX as usize {
                return -1;
            }
            let engine = unsafe { &mut *(ptr as *mut FfiEngine) };
            if count == 0 {
                if !out_cum_ofi.is_null() {
                    unsafe {
                        *out_cum_ofi = engine.ofi.cumulative_usd();
                    }
                }
                return 0;
            }
            if sides.is_null()
                || price_ticks.is_null()
                || quantities.is_null()
                || timestamps_ns.is_null()
            {
                return -1;
            }

            let s_slice = unsafe { slice::from_raw_parts(sides, count) };
            let p_slice = unsafe { slice::from_raw_parts(price_ticks, count) };
            let q_slice = unsafe { slice::from_raw_parts(quantities, count) };
            let t_slice = unsafe { slice::from_raw_parts(timestamps_ns, count) };

            let mut processed = 0_i32;
            for i in 0..count {
                let side = match s_slice[i] {
                    0 => Side::Bid,
                    1 => Side::Ask,
                    _ => return -2,
                };
                let delta = L2Delta {
                    side,
                    price_ticks: p_slice[i],
                    quantity: q_slice[i],
                    timestamp_ns: t_slice[i],
                };
                if engine.apply_l2(delta).is_err() {
                    return -3;
                }
                processed += 1;
            }

            if !out_cum_ofi.is_null() {
                unsafe {
                    *out_cum_ofi = engine.ofi.cumulative_usd();
                }
            }
            processed
        }))
        .unwrap_or(-1)
    }

    #[no_mangle]
    pub extern "C" fn mabuy_engine_observe_trade(
        ptr: *mut c_void,
        timestamp_ns: u64,
        price_usd: f64,
    ) -> i32 {
        catch_unwind(AssertUnwindSafe(|| {
            if ptr.is_null() {
                return -1;
            }
            if !price_usd.is_finite() || price_usd <= 0.0 {
                return -2;
            }
            let engine = unsafe { &mut *(ptr as *mut FfiEngine) };
            let _ = engine.observe_trade(timestamp_ns, price_usd);
            0
        }))
        .unwrap_or(-1)
    }

    #[no_mangle]
    pub extern "C" fn mabuy_engine_set_funding_rate(
        ptr: *mut c_void,
        funding_rate: f64,
    ) -> i32 {
        catch_unwind(AssertUnwindSafe(|| {
            if ptr.is_null() {
                return -1;
            }
            if !funding_rate.is_finite() {
                return -2;
            }
            let engine = unsafe { &mut *(ptr as *mut FfiEngine) };
            engine.set_funding_rate(funding_rate);
            0
        }))
        .unwrap_or(-1)
    }

    #[no_mangle]
    pub extern "C" fn mabuy_engine_clear_circuit_breaker(ptr: *mut c_void) -> i32 {
        catch_unwind(AssertUnwindSafe(|| {
            if ptr.is_null() {
                return -1;
            }
            let engine = unsafe { &*(ptr as *const FfiEngine) };
            if engine.risk.clear_circuit_breaker() {
                1
            } else {
                0
            }
        }))
        .unwrap_or(-1)
    }

    #[no_mangle]
    pub extern "C" fn mabuy_engine_snapshot(
        ptr: *mut c_void,
        out: *mut EngineSnapshotC,
    ) -> i32 {
        catch_unwind(AssertUnwindSafe(|| {
            if ptr.is_null() || out.is_null() {
                return -1;
            }
            let engine = unsafe { &*(ptr as *const FfiEngine) };
            let risk_snap = engine.risk.snapshot();
            let mid = engine.book.mid_price_usd();
            let hurst_est = engine.hurst.estimate();
            let vol_est = engine.volatility.volatility();

            unsafe {
                *out = EngineSnapshotC {
                    cumulative_ofi_usd: engine.ofi.cumulative_usd(),
                    last_ofi_increment_usd: engine.ofi.last_increment_usd(),
                    mid_price_usd: mid.unwrap_or(0.0),
                    has_mid_price: u8::from(mid.is_some()),
                    hurst: hurst_est.unwrap_or(0.0),
                    has_hurst: u8::from(hurst_est.is_some()),
                    volatility: vol_est.unwrap_or(0.0),
                    has_volatility: u8::from(vol_est.is_some()),
                    risk_state: risk_snap.state as u8,
                    hurst_regime: risk_snap.regime as u8,
                    bid_count: engine.book.bid_count(),
                    ask_count: engine.book.ask_count(),
                    generation: risk_snap.generation,
                };
            }
            0
        }))
        .unwrap_or(-1)
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

    #[test]
    fn ffi_hurst_multiscale_and_2scale_handle_trends_and_guards() {
        assert!((ffi::mabuy_hurst_rs_multiscale(std::ptr::null(), 100, 8) - 0.50).abs() < 1e-12);
        assert!((ffi::mabuy_hurst_2scale(std::ptr::null(), 100, 20) - 0.50).abs() < 1e-12);

        let short = [100.0; 10];
        assert!((ffi::mabuy_hurst_rs_multiscale(short.as_ptr(), short.len(), 8) - 0.50).abs() < 1e-12);
        assert!((ffi::mabuy_hurst_2scale(short.as_ptr(), short.len(), 20) - 0.50).abs() < 1e-12);

        // Trending AR(1) prices + non-positive/NaN entries that must be filtered
        let mut state = 0x1234_5678_9ABC_DEF0_u64;
        let mut next_noise = || -> f64 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (((state >> 11) as f64 / ((u64::MAX >> 11) as f64)) - 0.5) * 0.008
        };
        let mut prices = Vec::with_capacity(120);
        let mut r = 0.005_f64;
        let mut p = 100.0_f64;
        prices.push(p);
        for _ in 1..200 {
            r = 0.70 * r + next_noise();
            p *= r.exp();
            prices.push(p);
        }
        prices.push(-5.0);
        prices.push(0.0);
        prices.push(f64::NAN);
        prices.push(f64::INFINITY);

        let h_ms = ffi::mabuy_hurst_rs_multiscale(prices.as_ptr(), prices.len(), 8);
        let h_2s = ffi::mabuy_hurst_2scale(prices.as_ptr(), prices.len(), 20);
        assert!(h_ms > 0.55 && h_ms <= 0.95, "expected persistent h_ms > 0.55, got {h_ms}");
        assert!(h_2s > 0.55 && h_2s <= 0.95, "expected persistent h_2s > 0.55, got {h_2s}");

        // Oscillating mean-reverting series
        let mr_prices: Vec<f64> = (0..100).map(|i| if i % 2 == 0 { 100.0 } else { 101.5 }).collect();
        let h_mr_2s = ffi::mabuy_hurst_2scale(mr_prices.as_ptr(), mr_prices.len(), 20);
        assert!(h_mr_2s < 0.45, "expected mean-reverting h_mr_2s < 0.45, got {h_mr_2s}");
    }

    #[test]
    fn ffi_hurst_single_scale_and_rolling_volatility_match_kernels() {
        let mut out = 0.0_f64;
        assert_eq!(ffi::mabuy_hurst_single_scale(std::ptr::null(), 10, &mut out), -1);
        assert_eq!(ffi::mabuy_rolling_volatility(std::ptr::null(), 10, 4, &mut out), -1);

        let samples: Vec<f64> = (0..32).map(|i| ((i as f64) * 0.37).sin() * 0.01).collect();
        assert_eq!(ffi::mabuy_hurst_single_scale(samples.as_ptr(), 10, std::ptr::null_mut()), -1);
        assert_eq!(ffi::mabuy_rolling_volatility(samples.as_ptr(), 10, 4, std::ptr::null_mut()), -1);

        let mut kernel = HurstKernel::<32>::new();
        for &s in &samples {
            kernel.push(s);
        }
        let rc_h = ffi::mabuy_hurst_single_scale(samples.as_ptr(), samples.len(), &mut out);
        assert_eq!(rc_h, 1);
        assert!((out - kernel.estimate().unwrap()).abs() < 1e-12);

        let short = [0.01, 0.02, 0.03];
        assert_eq!(ffi::mabuy_hurst_single_scale(short.as_ptr(), short.len(), &mut out), 0);
        assert!(out.is_nan());

        let ret_with_nan = [0.01, f64::NAN, -0.01, 0.02, f64::INFINITY, -0.02, 0.0];
        let mut vol_k = RollingVolatility::<4>::new();
        for &v in &ret_with_nan {
            vol_k.push_return(v);
        }
        let rc_v = ffi::mabuy_rolling_volatility(ret_with_nan.as_ptr(), ret_with_nan.len(), 4, &mut out);
        assert_eq!(rc_v, 1);
        assert!((out - vol_k.volatility().unwrap()).abs() < 1e-12);

        assert_eq!(ffi::mabuy_rolling_volatility(short.as_ptr(), short.len(), 1, &mut out), 0);
        assert!(out.is_nan());
    }

    #[test]
    fn ffi_obi_snapshot_empty_walls_and_null_guards() {
        let mut snap = ffi::ObiSnapshotC::default();
        assert_eq!(
            ffi::mabuy_obi_snapshot(
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                0,
                150_000.0,
                std::ptr::null_mut()
            ),
            -1
        );
        assert_eq!(
            ffi::mabuy_obi_snapshot(
                std::ptr::null(),
                std::ptr::null(),
                1,
                std::ptr::null(),
                std::ptr::null(),
                0,
                150_000.0,
                &mut snap
            ),
            -1
        );

        // Empty book fallback
        assert_eq!(
            ffi::mabuy_obi_snapshot(
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                0,
                150_000.0,
                &mut snap
            ),
            0
        );
        assert_eq!(snap.obi_score, 0.0);
        assert_eq!(snap.bid_ask_ratio, 1.0);
        assert_eq!(snap.has_bid_wall, 0);
        assert_eq!(snap.has_ask_wall, 0);

        // Bid wall detection (50 / 100 >= 35% and 59000 * 50 >= 150000)
        let bp = [59000.0, 58900.0, 58800.0, 58700.0];
        let bq = [50.0, 15.0, 15.0, 20.0];
        let ap = [60100.0, 60200.0, 60300.0, 60400.0];
        let aq = [25.0, 25.0, 25.0, 25.0];
        assert_eq!(
            ffi::mabuy_obi_snapshot(
                bp.as_ptr(),
                bq.as_ptr(),
                bp.len(),
                ap.as_ptr(),
                aq.as_ptr(),
                ap.len(),
                150_000.0,
                &mut snap
            ),
            0
        );
        assert_eq!(snap.has_bid_wall, 1);
        assert_eq!(snap.has_bid_wall_price, 1);
        assert!((snap.bid_wall_price - 59000.0).abs() < 1e-12);
        assert_eq!(snap.has_ask_wall, 0);
        assert_eq!(snap.has_ask_wall_price, 0);
        assert!((snap.total_bid_vol - 100.0).abs() < 1e-12);
        assert!((snap.total_ask_vol - 100.0).abs() < 1e-12);
        assert!((snap.obi_score - 0.0).abs() < 1e-12);
    }

    #[test]
    fn ffi_stateful_engine_session_batch_l2_trades_and_guards() {
        assert!(ffi::mabuy_engine_new(0.0, 0.55, 0.03, 0.0005, 0.025).is_null());
        assert!(ffi::mabuy_engine_new(0.01, 0.55, -0.1, 0.0005, 0.025).is_null());
        assert!(ffi::mabuy_engine_new(0.01, 0.55, 0.03, 0.0005, 0.0).is_null());
        ffi::mabuy_engine_free(std::ptr::null_mut());

        let ptr = ffi::mabuy_engine_new(0.01, 0.55, 0.03, 0.0005, 0.05);
        assert!(!ptr.is_null());

        let sides = [0_u8, 1_u8];
        let ticks = [10_000_i64, 10_002_i64];
        let qtys = [2.0_f64, 1.0_f64];
        let ts = [1_u64, 2_u64];
        let mut cum_ofi = 0.0_f64;

        let rc_batch = ffi::mabuy_engine_apply_l2_batch(
            ptr,
            sides.as_ptr(),
            ticks.as_ptr(),
            qtys.as_ptr(),
            ts.as_ptr(),
            2,
            &mut cum_ofi,
        );
        assert_eq!(rc_batch, 2);
        // +100.0*2.0 - 100.02*1.0 = +99.98
        assert!((cum_ofi - 99.98).abs() < 1e-12);

        // Invalid side returns error without crashing
        let bad_sides = [9_u8];
        assert_eq!(
            ffi::mabuy_engine_apply_l2_batch(
                ptr,
                bad_sides.as_ptr(),
                ticks.as_ptr(),
                qtys.as_ptr(),
                ts.as_ptr(),
                1,
                &mut cum_ofi,
            ),
            -2
        );

        // Feed trending trades to transition into Trending -> Normal
        let mut price = 100.0_f64;
        let mut ret = 0.003_f64;
        for i in 0..40_u64 {
            assert_eq!(ffi::mabuy_engine_observe_trade(ptr, i + 10, price), 0);
            ret = 0.80 * ret + 0.0005 + ((i as f64) * 0.25).sin() * 0.0002;
            price *= ret.exp();
        }
        assert_eq!(ffi::mabuy_engine_observe_trade(ptr, 100, -1.0), -2);

        let mut snap = ffi::EngineSnapshotC::default();
        assert_eq!(ffi::mabuy_engine_snapshot(ptr, &mut snap), 0);
        assert_eq!(snap.has_mid_price, 1);
        assert!((snap.mid_price_usd - 100.01).abs() < 1e-12);
        assert_eq!(snap.has_hurst, 1);
        assert_eq!(snap.has_volatility, 1);
        assert_eq!(snap.bid_count, 1);
        assert_eq!(snap.ask_count, 1);
        assert_eq!(snap.hurst_regime, HurstRegime::Trending as u8);
        assert_eq!(snap.risk_state, RiskState::Normal as u8);

        assert_eq!(ffi::mabuy_engine_snapshot(std::ptr::null_mut(), &mut snap), -1);
        assert_eq!(ffi::mabuy_engine_snapshot(ptr, std::ptr::null_mut()), -1);
        ffi::mabuy_engine_free(ptr);
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
