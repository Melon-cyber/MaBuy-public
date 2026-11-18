/**
 * MaBuy Mobile API Client & WebSocket Streaming Service.
 * Implements exponential backoff reconnects and full REST integration.
 */

import {
  DashboardData,
  GatewaysResponse,
  PositionsResponse,
  SignalsResponse,
  RiskMatrixResponse,
  EmergencyKillResponse,
  StreamMessage
} from '../types';

const isBrowser = typeof window !== 'undefined';
const defaultHost = isBrowser ? `${window.location.protocol}//${window.location.hostname}:8000` : 'http://localhost:8000';
const defaultWsHost = isBrowser ? `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.hostname}:8000` : 'ws://localhost:8000';

export const DEFAULT_HOST = (typeof process !== 'undefined' && process.env?.REACT_APP_API_URL) || defaultHost;
export const DEFAULT_WS_HOST = (typeof process !== 'undefined' && process.env?.REACT_APP_WS_URL) || defaultWsHost;
export const DEFAULT_AUTH_TOKEN = (typeof process !== 'undefined' && process.env?.REACT_APP_AUTH_TOKEN) || (isBrowser && (window as any).__MABUY_TOKEN__) || '';

export class MobileApiClient {
  private baseUrl: string;
  private wsUrl: string;
  private token: string;
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private isConnecting = false;
  private listeners: ((msg: StreamMessage) => void)[] = [];

  constructor(
    baseUrl: string = DEFAULT_HOST,
    wsUrl: string = DEFAULT_WS_HOST,
    token: string = DEFAULT_AUTH_TOKEN
  ) {
    this.baseUrl = baseUrl;
    this.wsUrl = wsUrl;
    this.token = token;
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseUrl}/api/v1/mobile${endpoint}`;
    const headers = {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${this.token}`,
      ...options.headers,
    };

    const res = await fetch(url, { ...options, headers });
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`API Error [${res.status}]: ${errText || res.statusText}`);
    }
    return res.json() as Promise<T>;
  }

  // --- REST Queries ---

  public async getDashboard(): Promise<DashboardData> {
    return this.request<DashboardData>('/dashboard');
  }

  public async getGateways(): Promise<GatewaysResponse> {
    return this.request<GatewaysResponse>('/gateways');
  }

  public async getPositions(): Promise<PositionsResponse> {
    return this.request<PositionsResponse>('/positions');
  }

  public async getSignals(limit: number = 25): Promise<SignalsResponse> {
    return this.request<SignalsResponse>(`/signals?limit=${limit}`);
  }

  public async getRiskMatrix(): Promise<RiskMatrixResponse> {
    return this.request<RiskMatrixResponse>('/risk-matrix');
  }

  public async emergencyKill(scope: string = 'ALL'): Promise<EmergencyKillResponse> {
    return this.request<EmergencyKillResponse>('/emergency/kill', {
      method: 'POST',
      body: JSON.stringify({
        scope,
        action: 'CANCEL_ORDERS_AND_CLOSE_POSITIONS',
        confirmation_code: 'KILL-MABUY',
        reason: 'Mobile App Cockpit Guard triggered',
      }),
    });
  }

  // --- WebSocket Streaming ---

  public subscribeStream(callback: (msg: StreamMessage) => void): () => void {
    this.listeners.push(callback);
    if (!this.ws && !this.isConnecting) {
      this.connectWebSocket();
    }
    return () => {
      this.listeners = this.listeners.filter((l) => l !== callback);
    };
  }

  private connectWebSocket() {
    this.isConnecting = true;
    const wsUrl = `${this.wsUrl}/api/v1/mobile/stream?token=${this.token}`;
    
    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        this.isConnecting = false;
        this.reconnectAttempts = 0;
        console.log('Mobile WS connected to MaBuy Citadel');
      };

      this.ws.onmessage = (event) => {
        try {
          const data: StreamMessage = JSON.parse(event.data);
          this.listeners.forEach((l) => l(data));
        } catch (e) {
          console.warn('WS Message parse error:', e);
        }
      };

      this.ws.onclose = () => {
        this.ws = null;
        this.isConnecting = false;
        this.scheduleReconnect();
      };

      this.ws.onerror = (e) => {
        console.warn('WS Error:', e);
      };
    } catch (e) {
      this.isConnecting = false;
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect() {
    const delay = Math.min(1000 * Math.pow(1.5, this.reconnectAttempts), 15000);
    this.reconnectAttempts++;
    setTimeout(() => {
      if (this.listeners.length > 0) {
        this.connectWebSocket();
      }
    }, delay);
  }
}

export const mobileClient = new MobileApiClient();
