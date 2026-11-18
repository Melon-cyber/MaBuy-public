import React, { useState } from 'react';
import {
  SafeAreaView,
  StyleSheet,
  View,
  Text,
  TouchableOpacity,
  StatusBar
} from 'react-native';
import { CockpitScreen } from './src/screens/CockpitScreen';
import { PositionsScreen } from './src/screens/PositionsScreen';
import { SignalsScreen } from './src/screens/SignalsScreen';
import { KillSwitchScreen } from './src/screens/KillSwitchScreen';

type TabType = 'cockpit' | 'positions' | 'signals' | 'kill';

export default function App() {
  const [activeTab, setActiveTab] = useState<TabType>('cockpit');

  const renderActiveScreen = () => {
    switch (activeTab) {
      case 'cockpit':
        return <CockpitScreen />;
      case 'positions':
        return <PositionsScreen />;
      case 'signals':
        return <SignalsScreen />;
      case 'kill':
        return <KillSwitchScreen />;
      default:
        return <CockpitScreen />;
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#090A0F" />

      {/* Screen Body */}
      <View style={styles.screenBody}>
        {renderActiveScreen()}
      </View>

      {/* Bottom Navigation Bar */}
      <View style={styles.bottomNav}>
        <TouchableOpacity
          style={[styles.navTab, activeTab === 'cockpit' && styles.activeTab]}
          onPress={() => setActiveTab('cockpit')}
        >
          <Text style={[styles.navIcon, activeTab === 'cockpit' && styles.activeNavText]}>📊</Text>
          <Text style={[styles.navLabel, activeTab === 'cockpit' && styles.activeNavText]}>Cockpit</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.navTab, activeTab === 'positions' && styles.activeTab]}
          onPress={() => setActiveTab('positions')}
        >
          <Text style={[styles.navIcon, activeTab === 'positions' && styles.activeNavText]}>📈</Text>
          <Text style={[styles.navLabel, activeTab === 'positions' && styles.activeNavText]}>Positions</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.navTab, activeTab === 'signals' && styles.activeTab]}
          onPress={() => setActiveTab('signals')}
        >
          <Text style={[styles.navIcon, activeTab === 'signals' && styles.activeNavText]}>⚡</Text>
          <Text style={[styles.navLabel, activeTab === 'signals' && styles.activeNavText]}>Signals</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.navTab, activeTab === 'kill' && styles.activeTab]}
          onPress={() => setActiveTab('kill')}
        >
          <Text style={[styles.navIcon, activeTab === 'kill' && styles.activeNavText]}>🚨</Text>
          <Text style={[styles.navLabel, activeTab === 'kill' && styles.activeNavKillText]}>Kill Switch</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#090A0F',
  },
  screenBody: {
    flex: 1,
  },
  bottomNav: {
    flexDirection: 'row',
    backgroundColor: '#0D1017',
    borderTopWidth: 1,
    borderTopColor: '#171E2D',
    paddingVertical: 8,
    paddingHorizontal: 12,
  },
  navTab: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 4,
    borderRadius: 8,
  },
  activeTab: {
    backgroundColor: '#141A26',
  },
  navIcon: {
    fontSize: 18,
    marginBottom: 2,
  },
  navLabel: {
    fontSize: 10,
    fontWeight: '700',
    color: '#657088',
    letterSpacing: 0.5,
  },
  activeNavText: {
    color: '#00FF66',
  },
  activeNavKillText: {
    color: '#FF3366',
  },
});
