import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TextInput,
  TouchableOpacity,
  Alert,
  ActivityIndicator
} from 'react-native';
import { mobileClient } from '../api/client';
import { DashboardData, PositionsResponse } from '../types';

export const KillSwitchScreen: React.FC = () => {
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);
  const [positions, setPositions] = useState<PositionsResponse | null>(null);
  const [confirmInput, setConfirmInput] = useState('');
  const [executing, setExecuting] = useState(false);
  const [resultMessage, setResultMessage] = useState<string | null>(null);

  const loadData = async () => {
    try {
      const [d, p] = await Promise.all([
        mobileClient.getDashboard(),
        mobileClient.getPositions(),
      ]);
      setDashboard(d);
      setPositions(p);
    } catch (e) {
      console.warn('Error fetching kill switch state:', e);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleKillSwitch = async () => {
    if (confirmInput.trim() !== 'KILL-MABUY') {
      Alert.alert('Verification Failed', "Please type exactly 'KILL-MABUY' to confirm emergency action.");
      return;
    }

    setExecuting(true);
    setResultMessage(null);
    try {
      const res = await mobileClient.emergencyKill('ALL');
      setResultMessage(`✅ SUCCESS: ${res.message} (OpID: ${res.operation_id})`);
      setConfirmInput('');
      loadData();
    } catch (e: any) {
      setResultMessage(`❌ ERROR: ${e.message}`);
    } finally {
      setExecuting(false);
    }
  };

  const isConfirmed = confirmInput.trim() === 'KILL-MABUY';
  const openCount = positions?.positions.length || 0;

  return (
    <ScrollView style={styles.container}>
      <View style={styles.headerRow}>
        <View>
          <Text style={styles.screenTitle}>EMERGENCY KILL SWITCH</Text>
          <Text style={styles.screenSubtitle}>Fail-Closed Capital Protection Protocol</Text>
        </View>
        <View style={styles.statusBadge}>
          <Text style={styles.statusText}>GUARD ARMED</Text>
        </View>
      </View>

      {/* Impact Preview Card */}
      <View style={styles.impactCard}>
        <Text style={styles.impactTitle}>IMPACT PREVIEW</Text>
        <View style={styles.impactRow}>
          <Text style={styles.impactLabel}>Active Positions to Close:</Text>
          <Text style={styles.impactVal}>{openCount}</Text>
        </View>
        <View style={styles.impactRow}>
          <Text style={styles.impactLabel}>Open Algo & Limit Orders:</Text>
          <Text style={styles.impactVal}>ALL CANCELLED</Text>
        </View>
        <View style={styles.impactRow}>
          <Text style={styles.impactLabel}>Target Portfolio Equity:</Text>
          <Text style={styles.impactVal}>${dashboard?.equity_usd.toFixed(2) || '40.20'} USDT</Text>
        </View>
        <View style={styles.impactRow}>
          <Text style={styles.impactLabel}>Resulting Strategy State:</Text>
          <Text style={[styles.impactVal, { color: '#00FF66' }]}>100% FLAT IN CASH</Text>
        </View>
      </View>

      {/* Warning Box */}
      <View style={styles.warningBox}>
        <Text style={styles.warningTitle}>⚠️ CRITICAL INSTITUTIONAL ACTION</Text>
        <Text style={styles.warningDesc}>
          Triggering the Kill Switch will immediately dispatch Market orders with `reduceOnly=True` across Binance,
          Bybit, and OKX to completely flatten exposure and purge all pending Stop-Loss and Take-Profit orders.
        </Text>
      </View>

      {/* Confirmation Input */}
      <View style={styles.inputContainer}>
        <Text style={styles.inputLabel}>TYPE "KILL-MABUY" TO CONFIRM:</Text>
        <TextInput
          style={[styles.input, isConfirmed && styles.inputValid]}
          value={confirmInput}
          onChangeText={setConfirmInput}
          placeholder="KILL-MABUY"
          placeholderTextColor="#4B556D"
          autoCapitalize="characters"
        />
      </View>

      {/* Trigger Button */}
      <TouchableOpacity
        style={[styles.killButton, !isConfirmed && styles.killButtonDisabled]}
        disabled={!isConfirmed || executing}
        onPress={handleKillSwitch}
      >
        {executing ? (
          <ActivityIndicator color="#FFFFFF" />
        ) : (
          <Text style={styles.killButtonText}>🚨 EXECUTE EMERGENCY KILL SWITCH</Text>
        )}
      </TouchableOpacity>

      {resultMessage && (
        <View style={styles.resultBox}>
          <Text style={styles.resultText}>{resultMessage}</Text>
        </View>
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
  statusBadge: {
    backgroundColor: 'rgba(255, 51, 102, 0.15)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#FF3366',
  },
  statusText: {
    color: '#FF3366',
    fontWeight: '800',
    fontSize: 10,
  },
  impactCard: {
    backgroundColor: '#10141F',
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: '#1E2738',
    marginBottom: 16,
  },
  impactTitle: {
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 1,
    color: '#657088',
    marginBottom: 12,
  },
  impactRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#171E2D',
  },
  impactLabel: {
    color: '#8A92A6',
    fontSize: 12,
  },
  impactVal: {
    color: '#FFFFFF',
    fontWeight: '700',
    fontSize: 12,
  },
  warningBox: {
    backgroundColor: 'rgba(255, 170, 0, 0.08)',
    borderRadius: 12,
    padding: 14,
    borderWidth: 1,
    borderColor: '#FFAA00',
    marginBottom: 18,
  },
  warningTitle: {
    color: '#FFAA00',
    fontSize: 12,
    fontWeight: '800',
    marginBottom: 4,
  },
  warningDesc: {
    color: '#DCE1EB',
    fontSize: 11,
    lineHeight: 16,
  },
  inputContainer: {
    marginBottom: 18,
  },
  inputLabel: {
    color: '#8A92A6',
    fontSize: 11,
    fontWeight: '700',
    marginBottom: 8,
  },
  input: {
    backgroundColor: '#121622',
    borderWidth: 1,
    borderColor: '#1E2738',
    borderRadius: 10,
    padding: 14,
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '800',
    textAlign: 'center',
    letterSpacing: 2,
  },
  inputValid: {
    borderColor: '#00FF66',
    backgroundColor: 'rgba(0, 255, 102, 0.05)',
  },
  killButton: {
    backgroundColor: '#FF1744',
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: 'center',
    marginBottom: 16,
  },
  killButtonDisabled: {
    backgroundColor: '#2A1820',
    opacity: 0.5,
  },
  killButtonText: {
    color: '#FFFFFF',
    fontWeight: '900',
    fontSize: 13,
    letterSpacing: 1,
  },
  resultBox: {
    backgroundColor: '#10141F',
    borderRadius: 10,
    padding: 14,
    borderWidth: 1,
    borderColor: '#1E2738',
  },
  resultText: {
    color: '#DCE1EB',
    fontSize: 12,
    fontWeight: '600',
  },
});
