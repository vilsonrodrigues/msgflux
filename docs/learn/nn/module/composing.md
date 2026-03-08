# Composing Modules

## Sub-Modules

Modules can contain other modules:

```python
class Pipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.preprocessor = Preprocessor()
        self.analyzer = Analyzer()

    def forward(self, data):
        data = self.preprocessor(data)
        return self.analyzer(data)
```

Sub-modules are automatically tracked in the state dict.

See also: [ModuleDict](module-dict.md), [ModuleList](module-list.md), [Sequential](sequential.md)
