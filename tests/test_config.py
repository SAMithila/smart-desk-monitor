"""
Unit tests for configuration module.
"""

import tempfile
from pathlib import Path

import pytest

from objectSpace.config import (
    DetectorConfig,
    TrackerConfig,
    VideoConfig,
    OutputConfig,
    PipelineConfig,
    COCO_CLASSES,
    get_default_config,
)


class TestDetectorConfig:
    """Tests for DetectorConfig."""
    
    def test_default_values(self):
        config = DetectorConfig()
        assert config.model_name == "maskrcnn_resnet50_fpn"
        assert config.device == "auto"  # default is "auto"
        assert config.default_confidence == 0.3
    
    def test_get_threshold_default(self):
        config = DetectorConfig()
        # Unknown class should return default
        assert config.get_threshold(999) == 0.3
    
    def test_get_threshold_custom(self):
        config = DetectorConfig()
        # Person class has custom threshold
        assert config.get_threshold(1) == 0.6
    
    def test_custom_thresholds(self):
        config = DetectorConfig(class_thresholds={1: 0.9, 2: 0.5})
        assert config.get_threshold(1) == 0.9
        assert config.get_threshold(2) == 0.5


class TestTrackerConfig:
    """Tests for TrackerConfig."""
    
    def test_default_values(self):
        config = TrackerConfig()
        assert config.max_age == 8
        assert config.min_hits == 3
        assert config.iou_threshold == 0.3
        assert config.process_noise == 0.1
        assert config.measurement_noise == 5.0


class TestVideoConfig:
    """Tests for VideoConfig."""
    
    def test_default_resolution(self):
        config = VideoConfig()
        assert config.target_width == 1920
        assert config.target_height == 1080

    def test_supported_formats(self):
        config = VideoConfig()
        assert ".mp4" in config.supported_formats
        assert ".mkv" in config.supported_formats


class TestPipelineConfig:
    """Tests for PipelineConfig."""
    
    def test_nested_configs(self):
        config = get_default_config()
        assert isinstance(config.detector, DetectorConfig)
        assert isinstance(config.tracker, TrackerConfig)
    
    def test_yaml_roundtrip(self):
        """Test saving and loading config from YAML."""
        config = get_default_config()
        config.detector.default_confidence = 0.5
        config.tracker.max_age = 10
        
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            
            # Save
            config.to_yaml(yaml_path)
            assert yaml_path.exists()
            
            # Load
            loaded = PipelineConfig.from_yaml(yaml_path)
            assert loaded.detector.default_confidence == 0.5
            assert loaded.tracker.max_age == 10


class TestCOCOClasses:
    """Tests for COCO class definitions."""
    
    def test_has_common_classes(self):
        assert "person" in COCO_CLASSES.values()
        assert "car" in COCO_CLASSES.values()
    
    def test_class_count(self):
        assert len(COCO_CLASSES) == 81  # 80 objects + background
