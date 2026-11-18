import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  RefreshControl,
  ActivityIndicator
} from 'react-native';
import { mobileClient } from '../api/client';
import { PositionsResponse, PositionItem } from '../types';

export const PositionsScreen: React.FC = () => {
  const [data, setData] = useState<PositionsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchPositions = async () => {
    try {
      const res = await mobileClient.getPositions();
      setData(res);
    } catch (e) {
      console.warn('Error fetching positions:', e);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchPositions();
    const interval = setInterval(fetchPositions, 5000);
    return () => clearInterval(interval);
  }, []);

  const onRefresh = () => {
    setRefreshing(true);
    fetchPositions();
  };

  if (loading && !data) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#00FF66" />
        <Text style={styles.loadingText}>Reading Live Exchange Positions...</Text>
      </View>
    );
  }

  const positions = data?.positions || [];

  return (
    <ScrollView
      style={styles.container}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#00FF66" />}
    >
      <View style={styles.headerRow}>
        <Text style={styles.screenTitle}>LIVE POSITIONS & RISK ESCORT</Text>
        <Text style={styles.openCountBadge}>{positions.length} ACTIVE</Text>
      </View>

      {positions.length === 0 ? (
        <View style={styles.emptyCard}>
          <Text style={styles.emptyIcon}>🛡️</Text>
          <Text style={styles.emptyTitle}>ALL CAPITAL IN 100% USDT</Text>
          <Text style={styles.emptySubtitle}>
            Hurst noise veto active ($H &lt; 0.52) or awaiting high-conviction sniper setup (>= 75% confidence).
            No capital at risk.
          </Text>
        </View>
      ) : (
        positions.map((p) => {
          const isLong = p.side === 'LONG';
          const pnlColor = p.unrealized_pnl >= 0 ? '#00FF66' : '#FF3366';

          return (
            <View key={p.position_id} style={styles.posCard}>
              {/* Position Top Bar */}
              <View style={styles.posTopBar}>
                <View style={styles.posSymRow}>
                  <Text style={styles.posSymbol}>{p.symbol}</Text>
                  <View style={[styles.sideBadge, { backgroundColor: isLong ? 'rgba(0, 255, 102, 0.15)' : 'rgba(255, 51, 102, 0.15)' }]}>
                    <Text style={[styles.sideText, { color: isLong ? '#00FF66' : '#FF3366' }]}>
                      {p.side} {p.leverage}X
                    </Text>
                  </View>
                </View>
                <Text style={styles.exchangeBadge}>{p.exchange}</Text>
              </View>

              {/* Price Row */}
              <View style={styles.priceGrid}>
                <View style={styles.priceCol}>
                  <Text style={styles.priceLabel}>ENTRY PRICE</Text>
                  <Text style={styles.priceVal}>${p.entry_price.toLocaleString()}</Text>
                </View>
                <View style={styles.priceCol}>
                  <Text style={styles.priceLabel}>MARK PRICE</Text>
                  <Text style={styles.priceVal}>${p.mark_price.toLocaleString()}</Text>
                </View>
                <View style={styles.priceCol}>
                  <Text style={styles.priceLabel}>UNREALIZED PNL</Text>
                  <Text style={[styles.priceVal, { color: pnlColor, fontWeight: '900' }]}>
                    {p.unrealized_pnl >= 0 ? '+' : ''}${p.unrealized_pnl.toFixed(2)} ({p.pnl_pct.toFixed(2)}%)
                  </Text>
                </View>
              </View>

              {/* 4-Tier Dynamic Trailing Progress */}
              <View style={styles.trailingContainer}>
                <View style={styles.trailingHeader}>
                  <Text style={styles.trailingTitle}>4-TIER TRAILING STOP</Text>
                  <Text style={styles.trailingLabelBadge}>{p.trailing.label}</Text>
                </View>

                {/* Progress bar with 4 stages */}
                <View style={styles.progressBarBg}>
                  <View style={[styles.progressBarFill, { width: `${Math.min(100, (p.trailing.tier / 4) * 100)}%` }]} />
                </View>

                <View style={styles.tiersRow}>
                  <Text style={[styles.tierNum, p.trailing.tier >= 1 && styles.activeTier]}>T1: Protect</Text>
                  <Text style={[styles.tierNum, p.trailing.tier >= 2 && styles.activeTier]}>T2: Breakeven</Text>
                  <Text style={[styles.tierNum, p.trailing.tier >= 3 && styles.activeTier]}>T3: Lock +1.5%</Text>
                  <Text style={[styles.tierNum, p.trailing.tier >= 4 && styles.activeTier]}>T4: TP Run</Text>
                </View>
              </View>

              {/* Liquidation Buffer */}
              <View style={styles.liqFooter}>
                <Text style={styles.liqText}>
                  Liquidation: <Text style={styles.liqVal}>${p.liquidation_price.toLocaleString()}</Text>
                </Text>
                <View style={styles.liqBadge}>
                  <Text style={styles.liqBadgeText}>Buffer: {p.liquidation_distance_pct}% (SAFE)</Text>
                </View>
              </View>
            </View>
          );
        })
      )}

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
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 20,
    marginBottom: 16,
  },
  screenTitle: {
    fontSize: 13,
    fontWeight: '800',
    letterSpacing: 1.2,
    color: '#DCE1EB',
  },
  openCountBadge: {
    backgroundColor: 'rgba(0, 229, 255, 0.15)',
    color: '#00E5FF',
    fontWeight: '800',
    fontSize: 11,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
  },
  emptyCard: {
    backgroundColor: '#10141F',
    borderRadius: 16,
    padding: 30,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#1E2738',
    marginTop: 20,
  },
  emptyIcon: {
    fontSize: 36,
    marginBottom: 12,
  },
  emptyTitle: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '800',
    letterSpacing: 1,
  },
  emptySubtitle: {
    color: '#657088',
    fontSize: 12,
    textAlign: 'center',
    marginTop: 8,
    lineHeight: 18,
  },
  posCard: {
    backgroundColor: '#10141F',
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: '#1E2738',
    marginBottom: 16,
  },
  posTopBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  posSymRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  posSymbol: {
    fontSize: 18,
    fontWeight: '900',
    color: '#FFFFFF',
    marginRight: 8,
  },
  sideBadge: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
  },
  sideText: {
    fontSize: 11,
    fontWeight: '800',
  },
  exchangeBadge: {
    color: '#657088',
    fontSize: 11,
    fontWeight: '700',
  },
  priceGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 10,
    borderTopWidth: 1,
    borderBottomWidth: 1,
    borderColor: '#171E2D',
  },
  priceCol: {
    flex: 1,
  },
  priceLabel: {
    fontSize: 9,
    color: '#657088',
    fontWeight: '700',
    marginBottom: 3,
  },
  priceVal: {
    fontSize: 13,
    color: '#DCE1EB',
    fontWeight: '700',
  },
  trailingContainer: {
    marginTop: 12,
    backgroundColor: '#0C0E17',
    padding: 12,
    borderRadius: 10,
  },
  trailingHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  trailingTitle: {
    color: '#657088',
    fontSize: 10,
    fontWeight: '800',
  },
  trailingLabelBadge: {
    color: '#00FF66',
    fontSize: 10,
    fontWeight: '700',
  },
  progressBarBg: {
    height: 6,
    backgroundColor: '#1E2536',
    borderRadius: 3,
    overflow: 'hidden',
    marginBottom: 6,
  },
  progressBarFill: {
    height: '100%',
    backgroundColor: '#00FF66',
  },
  tiersRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  tierNum: {
    fontSize: 9,
    color: '#4B556D',
    fontWeight: '700',
  },
  activeTier: {
    color: '#00FF66',
  },
  liqFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 12,
  },
  liqText: {
    fontSize: 11,
    color: '#657088',
  },
  liqVal: {
    color: '#DCE1EB',
    fontWeight: '700',
  },
  liqBadge: {
    backgroundColor: 'rgba(0, 255, 102, 0.1)',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
  },
  liqBadgeText: {
    color: '#00FF66',
    fontSize: 10,
    fontWeight: '800',
  },
});
