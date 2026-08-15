"""
Unit Tests
 - Focus: isolated code
 - Purpose: validate code function
 - Scope: Function or method
 - Environment: isoslated python env
"""

from crud  import get_all_items

def test_get_all():
    response = get_all_items()
    assert response == {"msg": "Hello"}