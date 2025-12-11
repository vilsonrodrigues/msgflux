"""Tests for AutoParams metaclass functionality."""

from unittest.mock import Mock

from msgflux.auto import AutoParams
from msgflux.nn.modules import LM, Agent, Predictor


def test_autoparams_basic_example():
    """Test AutoParams with a basic example class."""

    class Model(metaclass=AutoParams):
        def __init__(self, learning_rate, batch_size, epochs):
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.epochs = epochs

    class MyModel(Model):
        learning_rate = 0.001
        batch_size = 32
        epochs = 100

    # Test with all defaults
    m = MyModel()
    assert m.learning_rate == 0.001
    assert m.batch_size == 32
    assert m.epochs == 100

    # Test with partial override
    m2 = MyModel(learning_rate=0.01, batch_size=64)
    assert m2.learning_rate == 0.01
    assert m2.batch_size == 64
    assert m2.epochs == 100  # Still uses default


def test_autoparams_with_agent():
    """Test that Agent works with AutoParams for optional parameters."""

    # Create a custom agent class with default parameters
    # Note: We only set params that don't conflict with Agent's internal state
    class MyAssistant(Agent):
        name = "assistant"
        response_mode = "plain_response"

    # Create a mock model
    mock_model = Mock()
    mock_model.model_type = "chat_completion"

    # Instantiate with defaults
    agent = MyAssistant(model=mock_model)

    # Check that defaults were applied
    assert agent.name == "assistant"

    # Test that AutoParams is working by checking _auto_params
    assert hasattr(MyAssistant, "_auto_params")
    assert "name" in MyAssistant._auto_params
    assert MyAssistant._auto_params["name"] == "assistant"


def test_autoparams_inheritance():
    """Test that AutoParams works with multi-level inheritance."""

    class BaseModel(metaclass=AutoParams):
        def __init__(self, a, b, c):
            self.a = a
            self.b = b
            self.c = c

    class MiddleModel(BaseModel):
        a = 1
        b = 2

    class FinalModel(MiddleModel):
        c = 3

    # FinalModel should inherit a and b from MiddleModel
    m = FinalModel()
    assert m.a == 1
    assert m.b == 2
    assert m.c == 3

    # Can override inherited values
    m2 = FinalModel(a=10, b=20)
    assert m2.a == 10
    assert m2.b == 20
    assert m2.c == 3


def test_autoparams_with_lm():
    """Test that LM works with AutoParams."""

    # LM only has one required param (model), so we can test with AutoParams
    mock_model = Mock()
    mock_model.model_type = "chat_completion"

    lm = LM(model=mock_model)
    assert lm.model == mock_model

    # Custom LM with autoparams
    class MyLM(LM):
        pass

    lm2 = MyLM(model=mock_model)
    assert lm2.model == mock_model


def test_autoparams_does_not_capture_methods():
    """Test that AutoParams only captures non-callable attributes."""

    class Model(metaclass=AutoParams):
        param1 = "value1"
        param2 = 42

        def method(self):
            return "method"

        @classmethod
        def class_method(cls):
            return "class_method"

        def __init__(self, param1, param2):
            self.param1 = param1
            self.param2 = param2

    # Methods should not be in _auto_params
    assert "method" not in Model._auto_params
    assert "class_method" not in Model._auto_params

    # Only non-callable attributes
    assert "param1" in Model._auto_params
    assert "param2" in Model._auto_params

    # Test instantiation
    m = Model()
    assert m.param1 == "value1"
    assert m.param2 == 42


def test_autoparams_does_not_capture_private_attrs():
    """Test that AutoParams ignores private/special attributes."""

    class Model(metaclass=AutoParams):
        public_param = "public"
        _private_param = "private"
        __special_param = "special"

        def __init__(self, public_param):
            self.public_param = public_param

    # Only public params should be captured
    assert "public_param" in Model._auto_params
    assert "_private_param" not in Model._auto_params
    assert "__special_param" not in Model._auto_params

    m = Model()
    assert m.public_param == "public"


def test_autoparams_class_attribute_access():
    """Test that _auto_params is accessible on the class."""

    class Model(metaclass=AutoParams):
        param1 = "value1"
        param2 = 42

        def __init__(self, param1, param2):
            self.param1 = param1
            self.param2 = param2

    # _auto_params should be accessible on the class
    assert hasattr(Model, "_auto_params")
    assert Model._auto_params == {"param1": "value1", "param2": 42}

    # Test instantiation
    m = Model()
    assert m.param1 == "value1"
    assert m.param2 == 42

    # Instances should also have access to _auto_params via the class
    assert m._auto_params == Model._auto_params


def test_autoparams_docstring_as_parameter():
    """Test that docstring can be used as a parameter value."""

    class BaseClass(metaclass=AutoParams):
        _autoparams_use_docstring_for = "description"

        def __init__(self, description=None):
            self.description = description

    class WithDocstring(BaseClass):
        """This is my description from docstring"""

    class WithExplicitDescription(BaseClass):
        """This docstring is ignored"""

        description = "Explicit description wins"

    class WithoutDocstring(BaseClass):
        pass

    # Test with docstring
    obj1 = WithDocstring()
    assert obj1.description == "This is my description from docstring"

    # Test with explicit description (takes precedence over docstring)
    obj2 = WithExplicitDescription()
    assert obj2.description == "Explicit description wins"

    # Test without docstring
    obj3 = WithoutDocstring()
    assert obj3.description is None


def test_autoparams_agent_docstring_as_description():
    """Test that Agent uses docstring as description."""

    class MyAgent(Agent):
        """My custom agent that helps with coding"""

    class MyAgentWithExplicit(Agent):
        """This docstring is ignored"""

        description = "Explicit description"

    # Create mock model
    mock_model = Mock()
    mock_model.model_type = "chat_completion"

    # Test with docstring
    agent1 = MyAgent(model=mock_model)
    assert agent1.description == "My custom agent that helps with coding"

    # Test with explicit description (takes precedence)
    agent2 = MyAgentWithExplicit(model=mock_model)
    assert agent2.description == "Explicit description"


def test_autoparams_classname_as_parameter():
    """Test that class name can be used as a parameter value."""

    class BaseClass(metaclass=AutoParams):
        _autoparams_use_classname_for = "name"

        def __init__(self, name):
            self.name = name

    class SuperAgent(BaseClass):
        pass

    class CustomAgent(BaseClass):
        name = "my_custom_name"

    # Test with class name
    obj1 = SuperAgent()
    assert obj1.name == "SuperAgent"

    # Test with explicit name (takes precedence)
    obj2 = CustomAgent()
    assert obj2.name == "my_custom_name"


def test_autoparams_agent_classname_as_name():
    """Test that Agent uses class name as name when not provided."""

    class SuperAgent(Agent):
        """A super agent"""

    class CustomAgent(Agent):
        """A custom agent"""

        name = "explicit_name"

    # Create mock model
    mock_model = Mock()
    mock_model.model_type = "chat_completion"

    # Test with class name
    agent1 = SuperAgent(model=mock_model)
    assert agent1.name == "SuperAgent"
    assert agent1.description == "A super agent"

    # Test with explicit name (takes precedence)
    agent2 = CustomAgent(model=mock_model)
    assert agent2.name == "explicit_name"
    assert agent2.description == "A custom agent"


def test_autoparams_agent_both_features():
    """Test Agent with both classname and docstring features."""

    class MyAwesomeAgent(Agent):
        """This agent is awesome at solving problems"""

    # Create mock model
    mock_model = Mock()
    mock_model.model_type = "chat_completion"

    agent = MyAwesomeAgent(model=mock_model)

    # Both features should work together
    assert agent.name == "MyAwesomeAgent"  # From class name
    assert (
        agent.description == "This agent is awesome at solving problems"
    )  # From docstring
