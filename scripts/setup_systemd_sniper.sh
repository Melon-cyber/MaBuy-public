#!/bin/bash
# ==============================================================================
# 🎯 MABUY V5.1 SNIPER SYSTEMD SERVICE SETUP & OLD SERVICE KILLER
# ==============================================================================

echo "🛑 1. STOPPING ALL OLD SERVICES AND PROCESSES..."
sudo systemctl stop mabuy.service 2>/dev/null
sudo systemctl stop mabuy-trader.service 2>/dev/null
sudo systemctl stop mabuy_daemon.service 2>/dev/null
sudo systemctl stop auto-trader.service 2>/dev/null
sudo systemctl stop trader.service 2>/dev/null
sudo systemctl disable mabuy.service 2>/dev/null
sudo systemctl disable mabuy-trader.service 2>/dev/null

# Kill any rogue background python processes
sudo pkill -9 -f "auto_trader_daemon" 2>/dev/null
sudo pkill -9 -f "src/infrastructure/execution" 2>/dev/null

echo "✅ All rogue background services stopped!"

# Get current project directory
PROJECT_DIR=$(pwd)
PYTHON_PATH=$(which python3)

echo "📂 Project Directory: $PROJECT_DIR"
echo "🐍 Python Path: $PYTHON_PATH"

mkdir -p logs

# Run integrity verification
echo ""
echo "🔍 2. RUNNING SNIPER INTEGRITY AUDIT..."
python3 scripts/verify_sniper_readiness.py

# Create modern systemd service
SERVICE_FILE="/etc/systemd/system/mabuy-sniper.service"

echo ""
echo "⚙️ 3. CREATING /etc/systemd/system/mabuy-sniper.service..."

sudo bash -c "cat > $SERVICE_FILE" << EOF
[Unit]
Description=MaBuy V5.1 Institutional Sniper Auto-Trader (BTC/ETH/SOL 3x Isolated)
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$PROJECT_DIR
ExecStart=$PYTHON_PATH scripts/auto_trader_daemon.py
Restart=always
RestartSec=10
StandardOutput=append:$PROJECT_DIR/logs/daemon.log
StandardError=append:$PROJECT_DIR/logs/daemon_error.log
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

sudo chmod 644 $SERVICE_FILE
sudo systemctl daemon-reload
sudo systemctl enable mabuy-sniper.service
sudo systemctl restart mabuy-sniper.service

echo ""
echo "=================================================================="
echo "🎯 MABUY SNIPER SYSTEMD SERVICE SUCCESSFULLY ACTIVATED!"
echo "=================================================================="
echo "• Status: sudo systemctl status mabuy-sniper.service"
echo "• Logs:   tail -f logs/daemon.log"
echo "• Stop:   sudo systemctl stop mabuy-sniper.service"
echo "=================================================================="
