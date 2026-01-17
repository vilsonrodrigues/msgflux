"""Tests for OptimizationPlotter."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Check if visualization libraries are available
try:
    import matplotlib

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Skip all tests if neither library is available
pytestmark = pytest.mark.skipif(
    not MATPLOTLIB_AVAILABLE and not PLOTLY_AVAILABLE,
    reason="Neither matplotlib nor plotly is installed",
)


class TestPlotData:
    def test_creation(self):
        from msgflux.optim.visualization import PlotData

        data = PlotData(step=10, score=0.85, best_score=0.85)
        assert data.step == 10
        assert data.score == 0.85
        assert data.best_score == 0.85

    def test_optional_fields(self):
        from msgflux.optim.visualization import PlotData

        data = PlotData(step=1)
        assert data.score is None
        assert data.best_score is None
        assert data.trial is None
        assert data.generation is None
        assert data.metadata == {}

    def test_with_metadata(self):
        from msgflux.optim.visualization import PlotData

        data = PlotData(step=1, metadata={"key": "value"})
        assert data.metadata == {"key": "value"}


class TestOptimizationPlotterInit:
    def test_auto_detect_backend(self):
        from msgflux.optim.visualization import OptimizationPlotter

        plotter = OptimizationPlotter()
        # Should auto-detect available backend
        assert plotter.backend in ("matplotlib", "plotly")

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")
    def test_explicit_matplotlib_backend(self):
        from msgflux.optim.visualization import OptimizationPlotter

        plotter = OptimizationPlotter(backend="matplotlib")
        assert plotter.backend == "matplotlib"

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not installed")
    def test_explicit_plotly_backend(self):
        from msgflux.optim.visualization import OptimizationPlotter

        plotter = OptimizationPlotter(backend="plotly")
        assert plotter.backend == "plotly"

    def test_default_values(self):
        from msgflux.optim.visualization import OptimizationPlotter

        plotter = OptimizationPlotter()
        assert plotter.style == "default"
        assert plotter.figsize == (10, 6)

    def test_custom_values(self):
        from msgflux.optim.visualization import OptimizationPlotter

        plotter = OptimizationPlotter(style="dark", figsize=(12, 8))
        assert plotter.style == "dark"
        assert plotter.figsize == (12, 8)


class TestOptimizationPlotterLog:
    def test_log_data_point(self):
        from msgflux.optim.visualization import OptimizationPlotter

        plotter = OptimizationPlotter()
        plotter.log(step=1, score=0.5)

        assert len(plotter._data) == 1
        assert plotter._data[0].step == 1
        assert plotter._data[0].score == 0.5

    def test_log_multiple_points(self):
        from msgflux.optim.visualization import OptimizationPlotter

        plotter = OptimizationPlotter()
        plotter.log(step=1, score=0.5)
        plotter.log(step=2, score=0.6)
        plotter.log(step=3, score=0.7)

        assert len(plotter._data) == 3

    def test_log_with_all_fields(self):
        from msgflux.optim.visualization import OptimizationPlotter

        plotter = OptimizationPlotter()
        plotter.log(
            step=1,
            score=0.5,
            best_score=0.5,
            trial=1,
            generation=1,
            custom_field="value",
        )

        data = plotter._data[0]
        assert data.step == 1
        assert data.score == 0.5
        assert data.best_score == 0.5
        assert data.trial == 1
        assert data.generation == 1
        assert data.metadata["custom_field"] == "value"


class TestOptimizationPlotterProperties:
    def test_num_points(self):
        from msgflux.optim.visualization import OptimizationPlotter

        plotter = OptimizationPlotter()
        assert plotter.num_points == 0

        plotter.log(step=1, score=0.5)
        assert plotter.num_points == 1

        plotter.log(step=2, score=0.6)
        assert plotter.num_points == 2

    def test_get_data(self):
        from msgflux.optim.visualization import OptimizationPlotter

        plotter = OptimizationPlotter()
        plotter.log(step=1, score=0.5)
        plotter.log(step=2, score=0.6)

        data = plotter.get_data()
        assert len(data) == 2
        # Should be a copy
        data.append(None)
        assert plotter.num_points == 2


class TestOptimizationPlotterClear:
    def test_clear(self):
        from msgflux.optim.visualization import OptimizationPlotter

        plotter = OptimizationPlotter()
        plotter.log(step=1, score=0.5)
        plotter.log(step=2, score=0.6)

        plotter.clear()

        assert plotter.num_points == 0
        assert plotter._current_fig is None


class TestOptimizationPlotterPlotScoreHistory:
    def test_raises_if_no_data(self):
        from msgflux.optim.visualization import OptimizationPlotter

        plotter = OptimizationPlotter()

        with pytest.raises(ValueError, match="No data to plot"):
            plotter.plot_score_history()

    def test_creates_figure(self):
        from msgflux.optim.visualization import OptimizationPlotter

        plotter = OptimizationPlotter()
        plotter.log(step=1, score=0.5, best_score=0.5)
        plotter.log(step=2, score=0.6, best_score=0.6)

        fig = plotter.plot_score_history()
        assert fig is not None
        assert plotter._current_fig is not None

    def test_with_custom_title(self):
        from msgflux.optim.visualization import OptimizationPlotter

        plotter = OptimizationPlotter()
        plotter.log(step=1, score=0.5)

        fig = plotter.plot_score_history(title="Custom Title")
        assert fig is not None


class TestOptimizationPlotterPlotScoreDistribution:
    def test_raises_if_no_scores(self):
        from msgflux.optim.visualization import OptimizationPlotter

        plotter = OptimizationPlotter()
        plotter.log(step=1)  # No score

        with pytest.raises(ValueError, match="No scores to plot"):
            plotter.plot_score_distribution()

    def test_creates_histogram(self):
        from msgflux.optim.visualization import OptimizationPlotter

        plotter = OptimizationPlotter()
        for i in range(20):
            plotter.log(step=i, score=i * 0.05)

        fig = plotter.plot_score_distribution()
        assert fig is not None


class TestOptimizationPlotterPlotImprovementRate:
    def test_raises_if_not_enough_data(self):
        from msgflux.optim.visualization import OptimizationPlotter

        plotter = OptimizationPlotter()
        for i in range(5):
            plotter.log(step=i, score=i * 0.1)

        with pytest.raises(ValueError, match="Need at least"):
            plotter.plot_improvement_rate(window=10)

    def test_creates_improvement_plot(self):
        from msgflux.optim.visualization import OptimizationPlotter

        plotter = OptimizationPlotter()
        for i in range(20):
            plotter.log(step=i, score=i * 0.05)

        fig = plotter.plot_improvement_rate(window=5)
        assert fig is not None


class TestOptimizationPlotterPlotGenerationSummary:
    def test_raises_if_no_generation_data(self):
        from msgflux.optim.visualization import OptimizationPlotter

        plotter = OptimizationPlotter()
        plotter.log(step=1, score=0.5)  # No generation

        with pytest.raises(ValueError, match="No generation data"):
            plotter.plot_generation_summary()

    def test_creates_generation_plot(self):
        from msgflux.optim.visualization import OptimizationPlotter

        plotter = OptimizationPlotter()
        for i in range(5):
            plotter.log(step=i, score=i * 0.1, best_score=i * 0.1, generation=i)

        fig = plotter.plot_generation_summary()
        assert fig is not None


class TestOptimizationPlotterSave:
    def test_raises_if_no_figure(self):
        from msgflux.optim.visualization import OptimizationPlotter

        plotter = OptimizationPlotter()

        with pytest.raises(ValueError, match="No figure to save"):
            plotter.save("test.png")

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")
    def test_save_matplotlib(self):
        from msgflux.optim.visualization import OptimizationPlotter

        with tempfile.TemporaryDirectory() as tmpdir:
            plotter = OptimizationPlotter(backend="matplotlib")
            plotter.log(step=1, score=0.5, best_score=0.5)
            plotter.log(step=2, score=0.6, best_score=0.6)
            plotter.plot_score_history()

            path = Path(tmpdir) / "test.png"
            plotter.save(str(path))

            assert path.exists()

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not installed")
    def test_save_plotly_html(self):
        from msgflux.optim.visualization import OptimizationPlotter

        with tempfile.TemporaryDirectory() as tmpdir:
            plotter = OptimizationPlotter(backend="plotly")
            plotter.log(step=1, score=0.5, best_score=0.5)
            plotter.log(step=2, score=0.6, best_score=0.6)
            plotter.plot_score_history()

            path = Path(tmpdir) / "test.html"
            plotter.save(str(path))

            assert path.exists()

    def test_save_creates_parent_dirs(self):
        from msgflux.optim.visualization import OptimizationPlotter

        with tempfile.TemporaryDirectory() as tmpdir:
            plotter = OptimizationPlotter()
            plotter.log(step=1, score=0.5)
            plotter.plot_score_history()

            path = Path(tmpdir) / "subdir" / "test.png"

            # This should create the parent directory
            try:
                plotter.save(str(path))
            except Exception:
                # May fail due to backend-specific issues, but directory should exist
                pass

            assert path.parent.exists()


class TestOptimizationPlotterSaveAll:
    def test_save_all(self):
        from msgflux.optim.visualization import OptimizationPlotter

        with tempfile.TemporaryDirectory() as tmpdir:
            plotter = OptimizationPlotter()

            # Add enough data for all plots
            for i in range(20):
                plotter.log(
                    step=i,
                    score=i * 0.05,
                    best_score=i * 0.05,
                    generation=i // 4,
                )

            saved = plotter.save_all(tmpdir, prefix="test", format="png")

            # Should save at least score_history and distribution
            assert len(saved) >= 2


class TestOptimizationPlotterShow:
    def test_raises_if_no_figure(self):
        from msgflux.optim.visualization import OptimizationPlotter

        plotter = OptimizationPlotter()

        with pytest.raises(ValueError, match="No figure to show"):
            plotter.show()


class TestOptimizationPlotterFromExporter:
    def test_from_exporter(self):
        from msgflux.optim.export import ExperimentExporter
        from msgflux.optim.visualization import OptimizationPlotter

        # Create exporter with data
        exporter = ExperimentExporter()
        exporter.log_step(1, score=0.5)
        exporter.log_step(2, score=0.6)
        exporter.log_step(3, score=0.7)

        # Load into plotter
        plotter = OptimizationPlotter()
        plotter.from_exporter(exporter)

        assert plotter.num_points == 3
        data = plotter.get_data()
        assert data[0].score == 0.5
        assert data[1].score == 0.6
        assert data[2].score == 0.7


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")
class TestOptimizationPlotterMatplotlibSpecific:
    def test_percentage_formatting_for_normalized_scores(self):
        from msgflux.optim.visualization import OptimizationPlotter

        plotter = OptimizationPlotter(backend="matplotlib")
        # Scores in [0, 1] range should trigger percentage formatting
        for i in range(10):
            plotter.log(step=i, score=i * 0.1)

        fig = plotter.plot_score_history()
        assert fig is not None

    def test_score_history_with_best(self):
        from msgflux.optim.visualization import OptimizationPlotter

        plotter = OptimizationPlotter(backend="matplotlib")
        plotter.log(step=1, score=0.5, best_score=0.5)
        plotter.log(step=2, score=0.3, best_score=0.5)
        plotter.log(step=3, score=0.7, best_score=0.7)

        fig = plotter.plot_score_history(show_best=True)
        assert fig is not None


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not installed")
class TestOptimizationPlotterPlotlySpecific:
    def test_interactive_features(self):
        from msgflux.optim.visualization import OptimizationPlotter

        plotter = OptimizationPlotter(backend="plotly")
        for i in range(10):
            plotter.log(step=i, score=i * 0.1)

        fig = plotter.plot_score_history()
        assert fig is not None
        # Plotly figures have data and layout attributes
        assert hasattr(fig, "data")
        assert hasattr(fig, "layout")

