/**
 * TypeScript Data Models & DTOs for MaBuy Mobile Client.
 * Synchronized with FastAPI Mobile Gateway (/api/v1/mobile).
 */

export interface DashboardData {
  status: string;
  timestamp: string;
  fund_name: string;
  architecture_tier: string;
  server_heartbeat_ms: number;
  balance_usd: number;
  equity_usd: number;
  free_margin_usd: number;
  pnl: {
    daily_usd: number;
    daily_percent: number;
    unrealized_usd: number;
    total_realized_usd: number;
    win_rate_pct: number;
    closed_trades_sample: number;
  };
  risk: {
    margin_ratio: number;
    risk_level: 'NORMAL' | 'ELEVATED' | 'CRITICAL';
    active_positions_count: number;
    circuit_breaker: string;
    liquidation_protection_buffer: string;
  };
  gateways_summary: {
    online: number;
    total: number;
    all_gates_passed: boolean;
    status: string;
  };
}

export interface GateInfo {
  id: string;
  name: string;
  status: 'ONLINE' | 'ACTIVE' | 'DEGRADED' | 'OFFLINE';
  latency_ms?: number;
  mode: string;
  details?: string;
}

export interface GatewaysResponse {
  status: string;
  timestamp: string;
  total_active: number;
  total_configured: number;
  gates: GateInfo[];
}

export interface PositionTrailing {
  tier: number;
  label: string;
  move_pct: number;
  stop_price: number;
}

export interface PositionItem {
  position_id: string;
  exchange: string;
  symbol: string;
  side: 'LONG' | 'SHORT';
  size: number;
  entry_price: number;
  mark_price: number;
  unrealized_pnl: number;
  pnl_pct: number;
  leverage: number;
  liquidation_price: number;
  liquidation_distance_pct: number;
  trailing: PositionTrailing;
}

export interface PositionsResponse {
  status: string;
  timestamp: string;
  total_open: number;
  positions: PositionItem[];
}

export interface SignalItem {
  id: number;
  time: string;
  symbol: string;
  direction: 'BULLISH' | 'BEARISH';
  confidence_pct: number;
  entry_price: number;
  exit_price?: number;
  pnl_usd?: number;
  pnl_formatted: string;
  status: 'OPEN' | 'CLOSED' | 'CANCELLED';
  close_reason?: string;
  hash_short: string;
  full_hash: string;
}

export interface SignalsResponse {
  status: string;
  timestamp: string;
  count: number;
  signals: SignalItem[];
}

export interface RiskMatrixResponse {
  status: string;
  timestamp: string;
  funding_rates: Record<string, { rate_8h_pct: string; regime: string }>;
  hurst_regimes: Record<string, { hurst: number; regime: string; min_conf_required: string }>;
  cross_asset_correlation: {
    BTC_ETH: number;
    BTC_SOL: number;
    haircut_applied: boolean;
    max_same_direction_limit: number;
  };
  circuit_breaker: {
    status: string;
    threshold_pct: string;
    consecutive_loss_penalty: boolean;
    cooling_off: boolean;
  };
}

export interface EmergencyKillResponse {
  status: string;
  operation_id: string;
  timestamp: string;
  action: string;
  scope: string;
  cancelled_orders_count: number;
  closed_positions_count: number;
  message: string;
}

export interface StreamMessage {
  type: 'heartbeat' | 'pnl' | 'alert' | 'position_update';
  sequence: number;
  timestamp: string;
  data: any;
}
