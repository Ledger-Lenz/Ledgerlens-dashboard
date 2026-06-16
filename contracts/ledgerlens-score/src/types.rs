use soroban_sdk::{contracttype, Address, Symbol};

/// On-chain risk score record for a (wallet, asset_pair) combination.
#[contracttype]
#[derive(Clone, Debug)]
pub struct RiskScore {
    /// Composite risk score 0–100. Higher = more suspicious.
    pub score: u32,
    /// True if the Benford's Law anomaly detector flagged this wallet/pair.
    pub benford_flag: bool,
    /// True if the ensemble ML classifier flagged this wallet/pair.
    pub ml_flag: bool,
    /// Model confidence 0–100.
    pub confidence: u32,
    /// Ledger timestamp (Unix seconds) of the last score update.
    pub timestamp: u64,
}

/// Admin-level contract configuration stored in instance storage.
#[contracttype]
#[derive(Clone, Debug)]
pub struct Config {
    pub admin: Address,
    /// Authorised score-submission service account.
    pub service_account: Address,
    /// Score threshold above which a wallet is considered flagged on-chain.
    pub alert_threshold: u32,
}

/// Storage key types for persistent and instance storage.
#[contracttype]
pub enum DataKey {
    Config,
    Score(Address, Symbol),   // (wallet, asset_pair)
    Flagged(Address),         // whether wallet is currently above threshold
}
