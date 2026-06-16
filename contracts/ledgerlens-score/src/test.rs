#![cfg(test)]

use soroban_sdk::{symbol_short, testutils::Address as _, Address, Env};

use crate::{errors::LedgerLensError, LedgerLensContract, LedgerLensContractClient};

fn setup() -> (Env, LedgerLensContractClient<'static>, Address, Address) {
    let env = Env::default();
    env.mock_all_auths();

    let contract_id = env.register_contract(None, LedgerLensContract);
    let client = LedgerLensContractClient::new(&env, &contract_id);

    let admin = Address::generate(&env);
    let service = Address::generate(&env);

    client.initialize(&admin, &service, &75);
    (env, client, admin, service)
}

#[test]
fn test_initialize_and_double_init() {
    let (_, client, admin, service) = setup();
    let result = client.try_initialize(&admin, &service, &75);
    assert_eq!(result, Err(Ok(LedgerLensError::AlreadyInitialized)));
}

#[test]
fn test_submit_and_get_score() {
    let (_, client, _, _) = setup();
    let wallet = Address::generate(&client.env);
    let pair = symbol_short!("XLMUSDC");

    client.submit_score(&wallet, &pair, &80, &true, &true, &90);
    let score = client.get_score(&wallet, &pair);

    assert_eq!(score.score, 80);
    assert!(score.benford_flag);
    assert!(score.ml_flag);
    assert_eq!(score.confidence, 90);
}

#[test]
fn test_is_flagged_above_threshold() {
    let (_, client, _, _) = setup();
    let wallet = Address::generate(&client.env);
    let pair = symbol_short!("XLMUSDC");

    assert!(!client.is_flagged(&wallet));
    client.submit_score(&wallet, &pair, &80, &true, &true, &90);
    assert!(client.is_flagged(&wallet));
}

#[test]
fn test_is_cleared_below_threshold() {
    let (_, client, _, _) = setup();
    let wallet = Address::generate(&client.env);
    let pair = symbol_short!("XLMUSDC");

    client.submit_score(&wallet, &pair, &80, &true, &true, &90);
    assert!(client.is_flagged(&wallet));

    client.submit_score(&wallet, &pair, &30, &false, &false, &95);
    assert!(!client.is_flagged(&wallet));
}

#[test]
fn test_score_not_found() {
    let (_, client, _, _) = setup();
    let wallet = Address::generate(&client.env);
    let pair = symbol_short!("XLMUSDC");

    let result = client.try_get_score(&wallet, &pair);
    assert_eq!(result, Err(Ok(LedgerLensError::ScoreNotFound)));
}

#[test]
fn test_invalid_score_rejected() {
    let (_, client, _, _) = setup();
    let wallet = Address::generate(&client.env);
    let pair = symbol_short!("XLMUSDC");

    let result = client.try_submit_score(&wallet, &pair, &101, &false, &false, &50);
    assert_eq!(result, Err(Ok(LedgerLensError::InvalidScore)));
}

#[test]
fn test_update_threshold() {
    let (_, client, _, _) = setup();
    let wallet = Address::generate(&client.env);
    let pair = symbol_short!("XLMUSDC");

    // Score 60 is below default threshold 75 — not flagged
    client.submit_score(&wallet, &pair, &60, &false, &false, &80);
    assert!(!client.is_flagged(&wallet));

    // Lower threshold to 50 — now 60 should flag
    client.update_threshold(&50);
    client.submit_score(&wallet, &pair, &60, &false, &false, &80);
    assert!(client.is_flagged(&wallet));
}

#[test]
fn test_invalid_threshold_rejected() {
    let (_, client, _, _) = setup();
    let result = client.try_update_threshold(&101);
    assert_eq!(result, Err(Ok(LedgerLensError::InvalidThreshold)));
}
