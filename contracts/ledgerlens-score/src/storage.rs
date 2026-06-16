use soroban_sdk::{Address, Env, Symbol};

use crate::types::{Config, DataKey, RiskScore};

const SCORE_TTL_LEDGERS: u32 = 518400; // ~30 days at ~5s per ledger

pub fn has_config(env: &Env) -> bool {
    env.storage().instance().has(&DataKey::Config)
}

pub fn get_config(env: &Env) -> Config {
    env.storage().instance().get(&DataKey::Config).unwrap()
}

pub fn set_config(env: &Env, config: &Config) {
    env.storage().instance().set(&DataKey::Config, config);
}

pub fn get_score(env: &Env, wallet: &Address, pair: &Symbol) -> Option<RiskScore> {
    let key = DataKey::Score(wallet.clone(), pair.clone());
    env.storage().persistent().get(&key)
}

pub fn set_score(env: &Env, wallet: &Address, pair: &Symbol, score: &RiskScore) {
    let key = DataKey::Score(wallet.clone(), pair.clone());
    env.storage().persistent().set(&key, score);
    env.storage()
        .persistent()
        .extend_ttl(&key, SCORE_TTL_LEDGERS, SCORE_TTL_LEDGERS);
}

pub fn is_flagged(env: &Env, wallet: &Address) -> bool {
    env.storage()
        .persistent()
        .get(&DataKey::Flagged(wallet.clone()))
        .unwrap_or(false)
}

pub fn set_flagged(env: &Env, wallet: &Address, flagged: bool) {
    let key = DataKey::Flagged(wallet.clone());
    env.storage().persistent().set(&key, &flagged);
    env.storage()
        .persistent()
        .extend_ttl(&key, SCORE_TTL_LEDGERS, SCORE_TTL_LEDGERS);
}
