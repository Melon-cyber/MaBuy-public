import sqlite3
import json
import datetime
import requests
import os
import schedule
import time
import subprocess
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("weekly_retrain")

# Configuration
DB_PATH = os.environ.get('MABUY_DB_PATH', '/root/mabuy2/data/mabuy_prod.db')
OPENAI_API_KEY = os.environ.get('RETRAIN_OPENAI_API_KEY')
# NOTE: Using gpt-4o as it is an officially supported and highly capable model for this task.
OPENAI_MODEL = os.environ.get('RETRAIN_ASTRA_MODEL', 'gpt-6-astra')
MODEL_WEIGHTS_PATH = os.environ.get('MABUY_MODELS_DIR', '/root/mabuy2/models/hyperparams.json')

def get_last_7_days_data() -> Dict[str, Any]:
    """Extracts trading data for the last 7 days from the database."""
    logger.info(f"Extracting 7-day data from {DB_PATH}")
    if not os.path.exists(DB_PATH):
        logger.warning(f"Database not found at {DB_PATH}. Using mock data for testing.")
        return {"error": "Database not found", "mock_data": True}
        
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        seven_days_ago = (datetime.datetime.utcnow() - datetime.timedelta(days=7)).isoformat()
        
        # Extract closed signals from the last 7 days
        cursor.execute('''
            SELECT symbol, direction, confidence, entry_price, exit_price, pnl_usd, close_reason
            FROM signal_log 
            WHERE status = 'CLOSED' AND created_at >= ?
        ''', (seven_days_ago,))
        
        rows = cursor.fetchall()
        data = [dict(row) for row in rows]
        
        conn.close()
        logger.info(f"Extracted {len(data)} trading records.")
        return {"trading_logs": data}
        
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        return {"error": str(e)}

def request_new_hyperparameters(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sends trading data to OpenAI API and requests new hyperparameters."""
    logger.info(f"Sending data to OpenAI using model: {OPENAI_MODEL}")
    
    if not OPENAI_API_KEY:
        logger.error("RETRAIN_OPENAI_API_KEY is not set.")
        return {}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    prompt = (
        "Analyze the following 7-day trading log for a quantitative trading bot. "
        "Based on the performance (pnl_usd, close_reasons, confidence), suggest updated hyperparameters "
        "for the ensemble model. Return ONLY a JSON object with the new parameters. "
        "Example format: {\"learning_rate\": 0.01, \"stop_loss_pct\": 0.02, \"take_profit_pct\": 0.05, \"confidence_threshold\": 0.75}"
    )

    payload = {
        "model": OPENAI_MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are a senior quantitative MLOps AI. Output strictly valid JSON."},
            {"role": "user", "content": f"{prompt}\n\nTrading Data:\n{json.dumps(data)}"}
        ]
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        content = result['choices'][0]['message']['content']
        new_params = json.loads(content)
        logger.info("Successfully received new hyperparameters from API.")
        return new_params
    except Exception as e:
        logger.error(f"Failed to get new hyperparameters: {e}")
        return {}

def update_local_weights(new_params: Dict[str, Any]) -> bool:
    """Safely updates the local hyperparameters file."""
    if not new_params:
        logger.warning("No new parameters to update.")
        return False
        
    logger.info(f"Updating local weights at {MODEL_WEIGHTS_PATH}")
    try:
        os.makedirs(os.path.dirname(MODEL_WEIGHTS_PATH), exist_ok=True)
        
        # Read existing to merge or just overwrite. We will overwrite for this script.
        with open(MODEL_WEIGHTS_PATH, 'w', encoding='utf-8') as f:
            json.dump(new_params, f, indent=4)
            
        logger.info("Local weights updated successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to update local weights: {e}")
        return False

def restart_daemon() -> None:
    """Gracefully restarts the mabuy-sniper systemd daemon."""
    logger.info("Restarting mabuy-sniper service...")
    try:
        # Use subprocess to run systemctl restart
        subprocess.run(['systemctl', 'restart', 'mabuy-sniper.service'], check=True, capture_output=True, text=True)
        logger.info("mabuy-sniper service restarted successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to restart mabuy-sniper service: {e.stderr}")
    except FileNotFoundError:
        logger.warning("systemctl command not found (likely not running on Linux/VPS). Skipping restart.")

def run_weekly_retrain_pipeline():
    """Main pipeline execution function."""
    logger.info("=== Starting Weekly Retrain Pipeline ===")
    
    # 1. Gather data
    data = get_last_7_days_data()
    
    if data and "trading_logs" in data and len(data["trading_logs"]) > 0:
        # 2. Request new parameters
        new_params = request_new_hyperparameters(data)
        
        # 3. Update weights
        if update_local_weights(new_params):
            # 4. Restart daemon
            restart_daemon()
    else:
        logger.warning("Not enough trading data to perform retraining or database error occurred.")
        
    logger.info("=== Weekly Retrain Pipeline Finished ===")

if __name__ == "__main__":
    logger.info("Weekly retrain script loaded.")
    logger.info("Scheduling retrain pipeline to run every Saturday at 00:00 (Midnight).")
    
    # Schedule to run every Saturday at midnight
    schedule.every().saturday.at("00:00").do(run_weekly_retrain_pipeline)
    
    # Optional: run once immediately for testing if a flag is passed
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--run-now':
        run_weekly_retrain_pipeline()
    else:
        logger.info("Entering schedule loop...")
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Exiting schedule loop.")
