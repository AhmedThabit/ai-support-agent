from app.agents.classifier import classify_ticket


def test_account():
    r = classify_ticket("I forgot my password and cannot login")
    assert r.category == "Account"


def test_billing():
    r = classify_ticket("My card was charged twice. Need a refund.")
    assert r.category == "Billing"


def test_bug():
    r = classify_ticket("The app shows a blank screen and throws an error")
    assert r.category == "Bug"
