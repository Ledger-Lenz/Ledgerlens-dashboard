#!/usr/bin/env bash
# Deploy ledgerlens-score Soroban contract to Stellar testnet or mainnet.
# Usage: ./deploy.sh [testnet|mainnet]
set -euo pipefail

NETWORK=${1:-testnet}
CONTRACT_DIR="$(cd "$(dirname "$0")/ledgerlens-score" && pwd)"
WASM_PATH="$CONTRACT_DIR/target/wasm32-unknown-unknown/release/ledgerlens_score.optimized.wasm"

if [[ "$NETWORK" == "mainnet" ]]; then
  RPC_URL="https://soroban-rpc.stellar.org"
  PASSPHRASE="Public Global Stellar Network ; September 2015"
else
  RPC_URL="https://soroban-testnet.stellar.org"
  PASSPHRASE="Test SDF Network ; September 2015"
fi

echo "▶  Network  : $NETWORK"
echo "▶  RPC URL  : $RPC_URL"

# 1. Build
echo ""
echo "── Building contract ─────────────────────────────────────────────────────"
cd "$CONTRACT_DIR"
cargo build --target wasm32-unknown-unknown --release

# 2. Optimise
echo ""
echo "── Optimising WASM ───────────────────────────────────────────────────────"
stellar contract optimize \
  --wasm "target/wasm32-unknown-unknown/release/ledgerlens_score.wasm"

# 3. Deploy
echo ""
echo "── Deploying to $NETWORK ──────────────────────────────────────────────────"
CONTRACT_ID=$(stellar contract deploy \
  --wasm "$WASM_PATH" \
  --source deployer \
  --network "$NETWORK" \
  --rpc-url "$RPC_URL" \
  --network-passphrase "$PASSPHRASE")

echo ""
echo "✅  Contract deployed: $CONTRACT_ID"

# 4. Write contract ID to .env
cd "$(git rev-parse --show-toplevel)"
if [[ -f .env ]]; then
  sed -i "s|^LEDGERLENS_CONTRACT_ID=.*|LEDGERLENS_CONTRACT_ID=$CONTRACT_ID|" .env
  echo "   Updated LEDGERLENS_CONTRACT_ID in .env"
fi

echo ""
echo "Next step — initialise the contract:"
echo "  stellar contract invoke --id $CONTRACT_ID \\"
echo "    --source <ADMIN_IDENTITY> --network $NETWORK -- \\"
echo "    initialize --admin <ADMIN_ADDR> --service_account <SERVICE_ADDR> --alert_threshold 75"
