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
import { SignalsResponse, SignalItem } from '../types';

export const SignalsScreen: React.FC = () => {
  const [data, setData] = useState<SignalsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchSignals = async () => {
    try {
      const res = await mobileClient.getSignals(30);
      setData(res);
    } catch (e) {
      console.warn('Error fetching signals:', e);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchSignals();
    const interval = setInterval(fetchSignals, 15000);
    return () => clearInterval(interval);
  }, []);

  const onRefresh = () => {
    setRefreshing(true);
    fetchSignals();
  };

  if (loading && !data) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#00FF66" />
        <Text style={styles.loadingText}>Reading Audited SQLite Signal Log...</Text>
      </View>
    );
  }

  const signals = data?.signals || [];

  return (
    <ScrollView
      style={styles.container}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#00FF66" />}
    >
      <View style={styles.headerRow}>
        <View>
          <Text style={styles.screenTitle}>ALPHA SIGNALS & AUDIT LOG</Text>
          <Text style={styles.screenSubtitle}>Cryptographic SHA-256 Provenance</Text>
        </View>
        <Text style={styles.countBadge}>{signals.length} AUDITED</Text>
      </View>

      {signals.map((sig) => {
        const isBull = sig.direction === 'BULLISH';
        const pnlColor = (sig.pnl_usd || 0) >= 0 ? '#00FF66' : '#FF3366';

        return (
          <View key={sig.id} style={styles.signalCard}>
            <View style={styles.cardTop}>
              <View style={styles.symGroup}>
                <Text style={styles.symbolText}>{sig.symbol}</Text>
                <View style={[styles.dirBadge, { backgroundColor: isBull ? 'rgba(0, 255, 102, 0.15)' : 'rgba(255, 51, 102, 0.15)' }]}>
                  <Text style={[styles.dirText, { color: isBull ? '#00FF66' : '#FF3366' }]}>
                    {sig.direction}
                  </Text>
                </View>
              </View>
              <Text style={styles.timeText}>{sig.time}</Text>
            </View>

            <View style={styles.metaRow}>
              <View style={styles.metaItem}>
                <Text style={styles.metaLabel}>ML CONFIDENCE</Text>
                <Text style={styles.confText}>{sig.confidence_pct}%</Text>
              </View>

              <View style={styles.metaItem}>
                <Text style={styles.metaLabel}>ENTRY / EXIT</Text>
                <Text style={styles.metaVal}>
                  ${sig.entry_price ? sig.entry_price.toLocaleString() : '-'} → ${sig.exit_price ? sig.exit_price.toLocaleString() : '-'}
                </Text>
              </View>

              <View style={styles.metaItem}>
                <Text style={styles.metaLabel}>RESULT PNL</Text>
                <Text style={[styles.pnlText, { color: pnlColor }]}>
                  {sig.pnl_formatted}
                </Text>
              </View>
            </View>

            <View style={styles.cardBottom}>
              <Text style={styles.hashLabel}>HASH: <Text style={styles.hashVal}>{sig.hash_short}</Text></Text>
              <View style={styles.statusBadge}>
                <Text style={styles.statusText}>{sig.status} ({sig.close_reason})</Text>
              </View>
            </View>
          </View>
        );
      })}

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
  screenSubtitle: {
    fontSize: 10,
    color: '#657088',
    marginTop: 2,
  },
  countBadge: {
    backgroundColor: '#151A26',
    color: '#00E5FF',
    fontWeight: '800',
    fontSize: 11,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
  },
  signalCard: {
    backgroundColor: '#10141F',
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    borderColor: '#1E2738',
    marginBottom: 12,
  },
  cardTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  symGroup: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  symbolText: {
    fontSize: 16,
    fontWeight: '800',
    color: '#FFFFFF',
    marginRight: 8,
  },
  dirBadge: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 6,
  },
  dirText: {
    fontSize: 10,
    fontWeight: '800',
  },
  timeText: {
    fontSize: 11,
    color: '#657088',
  },
  metaRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderTopWidth: 1,
    borderBottomWidth: 1,
    borderColor: '#171E2D',
  },
  metaItem: {
    flex: 1,
  },
  metaLabel: {
    fontSize: 9,
    color: '#657088',
    fontWeight: '700',
    marginBottom: 3,
  },
  confText: {
    fontSize: 13,
    fontWeight: '800',
    color: '#00E5FF',
  },
  metaVal: {
    fontSize: 11,
    color: '#DCE1EB',
    fontWeight: '600',
  },
  pnlText: {
    fontSize: 13,
    fontWeight: '800',
  },
  cardBottom: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 8,
  },
  hashLabel: {
    fontSize: 10,
    color: '#4B556D',
  },
  hashVal: {
    fontFamily: 'monospace',
    color: '#707C96',
  },
  statusBadge: {
    backgroundColor: '#151A26',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  statusText: {
    fontSize: 9,
    color: '#8A92A6',
    fontWeight: '700',
  },
});
