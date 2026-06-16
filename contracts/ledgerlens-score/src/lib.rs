#![no_std]

mod errors;
mod events;
mod storage;
mod types;

use soroban_sdk::{contract, contractimpl, Address, Env, Symbol};

use errors::LedgerLensError;
use storage::{get_config, get_score, has_config, is_flagged, set_config, set_flagged, set_score};
use types::{Config, RiskScore};

#[contract]
pub struct LedgerLensContract;

#[contractimpl]
impl LedgerLensContract {
    /// One-time contract initialisation. Sets admin, service account, and alert threshold.
    pub fn initialize(
        env: Env,
        admin: Address,
        service_account: Address,
        alert_threshold: u32,
    ) -> Result<(), LedgerLensError> {
        if has_config(&env) {
            return Err(LedgerLensError::AlreadyInitialized);
        }
        if alert_threshold > 100 {
            return Err(LedgerLensError::InvalidThreshold);
        }
        admin.require_auth();
        set_config(
            &env,
            &Config {
                admin,
                service_account,
                alert_threshold,
            },
        );
        Ok(())
    }

    /// Submit a computed risk score for a wallet / asset pair.
    /// Only the authorised service account may call this.
    pub fn submit_score(
        env: Env,
        wallet: Address,
        asset_pair: Symbol,
        score: u32,
        benford_flag: bool,
        ml_flag: bool,
        confidence: u32,
    ) -> Result<(), LedgerLensError> {
        let config = Self::require_init(&env)?;
        config.service_account.require_auth();

        if score > 100 || confidence > 100 {
            return Err(LedgerLensError::InvalidScore);
        }

        let risk = RiskScore {
            score,
            benford_flag,
            ml_flag,
            confidence,
            timestamp: env.ledger().timestamp(),
        };

        set_score(&env, &wallet, &asset_pair, &risk);
        events::score_submitted(&env, &wallet, &asset_pair, score);

        let threshold = config.alert_threshold;
        let currently_flagged = is_flagged(&env, &wallet);
        if score >= threshold && !currently_flagged {
            set_flagged(&env, &wallet, true);
            events::wallet_flagged(&env, &wallet, score);
        } else if score < threshold && currently_flagged {
            set_flagged(&env, &wallet, false);
            events::wallet_cleared(&env, &wallet);
        }

        Ok(())
    }

    /// Retrieve the current risk score for a wallet / asset pair.
    pub fn get_score(
        env: Env,
        wallet: Address,
        asset_pair: Symbol,
    ) -> Result<RiskScore, LedgerLensError> {
        Self::require_init(&env)?;
        get_score(&env, &wallet, &asset_pair).ok_or(LedgerLensError::ScoreNotFound)
    }

    /// Check whether a wallet is currently flagged above the alert threshold.
    pub fn is_flagged(env: Env, wallet: Address) -> bool {
        is_flagged(&env, &wallet)
    }

    /// Update the alert threshold (admin only).
    pub fn update_threshold(env: Env, new_threshold: u32) -> Result<(), LedgerLensError> {
        let mut config = Self::require_init(&env)?;
        config.admin.require_auth();
        if new_threshold > 100 {
            return Err(LedgerLensError::InvalidThreshold);
        }
        let old = config.alert_threshold;
        config.alert_threshold = new_threshold;
        set_config(&env, &config);
        events::threshold_updated(&env, old, new_threshold);
        Ok(())
    }

    /// Rotate the authorised service account (admin only).
    pub fn rotate_service_account(
        env: Env,
        new_account: Address,
    ) -> Result<(), LedgerLensError> {
        let mut config = Self::require_init(&env)?;
        config.admin.require_auth();
        config.service_account = new_account;
        set_config(&env, &config);
        Ok(())
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn require_init(env: &Env) -> Result<Config, LedgerLensError> {
        if !has_config(env) {
            return Err(LedgerLensError::NotInitialized);
        }
        Ok(get_config(env))
    }
}

#[cfg(test)]
mod test;
