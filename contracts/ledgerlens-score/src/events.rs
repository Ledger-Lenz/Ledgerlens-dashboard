use soroban_sdk::{symbol_short, Address, Env, Symbol};

pub fn score_submitted(env: &Env, wallet: &Address, pair: &Symbol, score: u32) {
    env.events().publish(
        (symbol_short!("score_sub"), wallet.clone()),
        (pair.clone(), score),
    );
}

pub fn wallet_flagged(env: &Env, wallet: &Address, score: u32) {
    env.events().publish(
        (symbol_short!("flagged"), wallet.clone()),
        score,
    );
}

pub fn wallet_cleared(env: &Env, wallet: &Address) {
    env.events().publish(
        (symbol_short!("cleared"), wallet.clone()),
        (),
    );
}

pub fn threshold_updated(env: &Env, old: u32, new: u32) {
    env.events().publish(
        symbol_short!("thr_upd"),
        (old, new),
    );
}
