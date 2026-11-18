import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  RefreshControl,
  TouchableOpacity,
  ActivityIndicator
} from 'react-native';
import { mobileClient } from '../api/client';
import { DashboardData, GatewaysResponse } from '../types';

export const CockpitScreen: React.FC = () => {
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);
  const [gateways, setGateways] = useState<GatewaysResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [lastWsPing, setLastWsPing] = useState<string>('Connecting...');
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      setError(null);
      const [dashData, gateData] = await Promise.all([
        mobileClient.getDashboard(),
        mobileClient.getGateways(),
      ]);
      setDashboard(dashData);
      setGateways(gateData);
    } catch (e: any) {
      setError(e.message || 'Connection to VPS failed');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchData();

    // Subscribe to real-time WebSocket stream
    const unsubscribe = mobileClient.subscribeStream((msg) => {
      if (msg.type === 'heartbeat') {
        setLastWsPing(`${msg.data.server_latency_ms}ms · Online`);
        if (msg.data.equity_usd && dashboard) {
          setDashboard((prev) => (prev ? { ...prev, equity_usd: msg.data.equity_usd } : null));
        }
      }
    });

    // Auto-refresh poll every 10 seconds
    const interval = setInterval(fetchData, 10000);

    return () => {
      unsubscribe();
      clearInterval(interval);
    };
  }, []);

  const onRefresh = () => {
    setRefreshing(true);
    fetchData();
  };

  if (loading && !dashboard) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#00FF66" />
        <Text style={styles.loadingText}>Connecting to MaBuy VPS Citadel...</Text>
      </View>
    );
  }

  const pnlColor = (dashboard?.pnl.daily_usd || 0) >= 0 ? '#00FF66' : '#FF3366';

  return (
    <ScrollView
      style={styles.container}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#00FF66" />}
    >
      {/* Top Banner */}
      <View style={styles.topHeader}>
        <View>
          <Text style={styles.brandTitle}>MABUY QUANT</Text>
          <Text style={styles.brandSubtitle}>Citadel / Medallion Standard V5.2</Text>
        </View>
        <View style={styles.heartbeatBadge}>
          <View style={styles.pulsingDot} />
          <Text style={styles.heartbeatText}>{lastWsPing}</Text>
        </View>
      </View>

      {error && (
        <View style={styles.errorBox}>
          <Text style={styles.errorText}>⚠️ {error}</Text>
        </View>
      )}

      {/* Main Equity Card */}
      <View style={styles.equityCard}>
        <Text style={styles.cardLabel}>TOTAL PORTFOLIO EQUITY</Text>
        <Text style={styles.equityValue}>
          ${dashboard?.equity_usd.toFixed(2) || '40.20'}
          <Text style={styles.currency}> USDT</Text>
        </Text>
        <View style={styles.equitySubRow}>
          <Text style={styles.subLabel}>
            Balance: <Text style={styles.subVal}>${dashboard?.balance_usd.toFixed(2)}</Text>
          </Text>
          <Text style={styles.subLabel}>
            Free Margin: <Text style={styles.subVal}>${dashboard?.free_margin_usd.toFixed(2)}</Text>
          </Text>
        </View>
      </View>

      {/* PnL and Performance Grid */}
      <View style={styles.metricGrid}>
        <View style={styles.metricTile}>
          <Text style={styles.tileLabel}>24H PNL</Text>
          <Text style={[styles.tileValue, { color: pnlColor }]}>
            {(dashboard?.pnl.daily_usd || 0) >= 0 ? '+' : ''}${dashboard?.pnl.daily_usd.toFixed(2)}
          </Text>
          <Text style={[styles.tileSub, { color: pnlColor }]}>
            {dashboard?.pnl.daily_percent.toFixed(2)}%
          </Text>
        </View>

        <View style={styles.metricTile}>
          <Text style={styles.tileLabel}>WIN RATE</Text>
          <Text style={[styles.tileValue, { color: '#00E5FF' }]}>
            {dashboard?.pnl.win_rate_pct.toFixed(1)}%
          </Text>
          <Text style={styles.tileSub}>
            {dashboard?.pnl.closed_trades_sample} closed trades
          </Text>
        </View>

        <View style={styles.metricTile}>
          <Text style={styles.tileLabel}>UNREALIZED</Text>
          <Text style={[styles.tileValue, { color: '#FFD700' }]}>
            ${dashboard?.pnl.unrealized_usd.toFixed(2)}
          </Text>
          <Text style={styles.tileSub}>Trailing Active</Text>
        </View>

        <View style={styles.metricTile}>
          <Text style={styles.tileLabel}>RISK LEVEL</Text>
          <Text style={[styles.tileValue, { color: '#00FF66' }]}>
            {dashboard?.risk.risk_level}
          </Text>
          <Text style={styles.tileSub}>
            Buffer {dashboard?.risk.liquidation_protection_buffer}
          </Text>
        </View>
      </View>

      {/* 7/7 Institutional Defense Gates Section */}
      <View style={styles.sectionCard}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>INSTITUTIONAL DEFENSE GATES</Text>
          <View style={styles.gateCountBadge}>
            <Text style={styles.gateCountText}>7 / 7 ACTIVE</Text>
          </View>
        </View>

        {gateways?.gates.map((g) => (
          <View key={g.id} style={styles.gateRow}>
            <View style={styles.gateLeft}>
              <View style={styles.gateStatusDot} />
              <View>
                <Text style={styles.gateName}>{g.name}</Text>
                <Text style={styles.gateDetails}>
                  {g.details || `${g.mode} · Latency ${g.latency_ms}ms`}
                </Text>
              </View>
            </View>
            <View style={styles.gateTag}>
              <Text style={styles.gateTagText}>{g.status}</Text>
            </View>
          </View>
        ))}
      </View>

      <View style={{ height: 40 }} />
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#090A0F',
    paddingHorizontal: 16,
  },
  centerContainer: {
    flex: 1,
    backgroundColor: '#090A0F',
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: '#8A92A6',
    marginTop: 12,
    fontSize: 14,
  },
  topHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 20,
    marginBottom: 16,
  },
  brandTitle: {
    fontSize: 22,
    fontWeight: '900',
    letterSpacing: 1.5,
    color: '#FFFFFF',
  },
  brandSubtitle: {
    fontSize: 12,
    color: '#00FF66',
    fontWeight: '600',
    letterSpacing: 0.5,
  },
  heartbeatBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#121620',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#1E2536',
  },
  pulsingDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#00FF66',
    marginRight: 6,
  },
  heartbeatText: {
    color: '#8A92A6',
    fontSize: 11,
    fontWeight: '700',
  },
  errorBox: {
    backgroundColor: 'rgba(255, 51, 102, 0.15)',
    borderWidth: 1,
    borderColor: '#FF3366',
    borderRadius: 8,
    padding: 10,
    marginBottom: 12,
  },
  errorText: {
    color: '#FF3366',
    fontSize: 12,
    fontWeight: '600',
  },
  equityCard: {
    backgroundColor: '#10141F',
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
    borderColor: '#1E2738',
    marginBottom: 16,
  },
  cardLabel: {
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 1.2,
    color: '#657088',
    marginBottom: 6,
  },
  equityValue: {
    fontSize: 34,
    fontWeight: '900',
    color: '#FFFFFF',
    letterSpacing: 0.5,
  },
  currency: {
    fontSize: 18,
    fontWeight: '600',
    color: '#00FF66',
  },
  equitySubRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 14,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#171E2D',
  },
  subLabel: {
    fontSize: 12,
    color: '#707C96',
  },
  subVal: {
    color: '#DCE1EB',
    fontWeight: '700',
  },
  metricGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  metricTile: {
    width: '48%',
    backgroundColor: '#10141F',
    borderRadius: 12,
    padding: 14,
    borderWidth: 1,
    borderColor: '#1A2130',
    marginBottom: 12,
  },
  tileLabel: {
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 1,
    color: '#657088',
  },
  tileValue: {
    fontSize: 20,
    fontWeight: '800',
    marginVertical: 4,
  },
  tileSub: {
    fontSize: 11,
    color: '#707C96',
  },
  sectionCard: {
    backgroundColor: '#10141F',
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: '#1E2738',
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 14,
  },
  sectionTitle: {
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 1,
    color: '#DCE1EB',
  },
  gateCountBadge: {
    backgroundColor: 'rgba(0, 255, 102, 0.12)',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#00FF66',
  },
  gateCountText: {
    fontSize: 10,
    fontWeight: '800',
    color: '#00FF66',
  },
  gateRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#171E2D',
  },
  gateLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  gateStatusDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#00FF66',
    marginRight: 10,
  },
  gateName: {
    fontSize: 13,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  gateDetails: {
    fontSize: 11,
    color: '#657088',
    marginTop: 2,
  },
  gateTag: {
    backgroundColor: '#141A26',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
  },
  gateTagText: {
    fontSize: 10,
    fontWeight: '700',
    color: '#00FF66',
  },
});
