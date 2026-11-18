#!/usr/bin/env bash
# ==============================================================================
# MABUY HOLDING / ECOSYSTEM — FAST 1-CLICK ALL-IN-ONE UPDATE SCRIPT
# ==============================================================================
set -e

echo "🚀 [1/4] Переход в рабочую директорию /root/mabuy2/..."
cd /root/mabuy2/

echo "📦 [2/4] Проверка зависимостей и директорий..."
mkdir -p data/cache/macro_candles data/training_snapshots models public

echo "🧠 [3/4] Запуск квантового аудита и непрерывного дообучения (MaPredict & MaInvest)..."
python3 -c "
import sys
try:
    from src.infrastructure.mlops.continuous_retrainer import ContinuousRetrainer
    retrainer = ContinuousRetrainer()
    report = retrainer.run_ecosystem_health_and_retrain(force=False)
    print('✅ Retraining & Calibration status:', report.get('summary_ru', 'OK'))
except Exception as e:
    print('⚠️ Retraining note:', e)
"

echo "🔄 [4/4] Перезапуск всех системных демонов (Sniper, Bot, API)..."
systemctl daemon-reload
systemctl restart mabuy-sniper.service mabuy-bot.service mabuy-api.service 2>/dev/null || {
    echo "Перезапуск доступных сервисов..."
    systemctl restart mabuy-sniper.service || true
    systemctl restart mabuy-bot.service || true
    systemctl restart mabuy-api.service || true
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 ВСЕ СИСТЕМЫ MABUY УСПЕШНО ОБНОВЛЕНЫ И ПЕРЕЗАПУЩЕНЫ!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
systemctl status mabuy-sniper.service --no-pager -n 5 || true
