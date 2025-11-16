#!/usr/bin/env python3
"""
Hybrid 1X2 Evaluation Pipeline

Comprehensive evaluation of 1X2 prediction models:
- ML 1X2 (existing)
- Scoreline-derived 1X2
- Hybrid 1X2 (new)
- Poisson/Elo 1X2 (optional)

ADDITIVE - does not modify existing models or pipelines.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.utils import setup_logging
from core.hybrid_1x2_metrics import Hybrid1X2Evaluator
from api.improved_prediction_service import ImprovedPredictionService
from core.data_loader import ESPNDataLoader


class Hybrid1X2EvaluationPipeline:
    """
    Comprehensive evaluation pipeline for 1X2 models
    
    Evaluates and compares multiple 1X2 prediction approaches
    on historical match data.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize evaluation pipeline"""
        default_config = self._get_default_config()
        if config:
            default_config.update(config)
        self.config = default_config
        self.logger = setup_logging()
        
        # Initialize components
        self.evaluator = Hybrid1X2Evaluator()
        self.prediction_service = None
        self.data_loader = ESPNDataLoader()
        
        # Results storage
        self.results = {}
        self.predictions = {}
        self.evaluation_data = None
        
        # Create output directories
        self._create_output_dirs()
        
        self.logger.info("🎯 Hybrid 1X2 Evaluation Pipeline initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'sample_size': 1000,  # Number of matches to evaluate
            'test_split': 0.2,    # Fraction for test set
            'random_seed': 42,
            'models_to_evaluate': ['ml', 'scoreline', 'hybrid', 'poisson'],
            'output_dir': 'reports/hybrid_1x2',
            'save_plots': True,
            'plot_format': 'png',
            'plot_dpi': 300
        }
    
    def _create_output_dirs(self):
        """Create output directories"""
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (output_dir / 'plots').mkdir(exist_ok=True)
        (output_dir / 'reports').mkdir(exist_ok=True)
    
    def load_evaluation_data(self) -> pd.DataFrame:
        """
        Load and prepare evaluation dataset
        
        Returns:
            DataFrame with match data for evaluation
        """
        try:
            self.logger.info("📂 Loading evaluation dataset...")
            
            # Load historical matches
            df = self.data_loader.load_fixtures()
            
            if df is None or df.empty:
                raise ValueError("No evaluation data available")
            
            self.logger.info(f"📊 Loaded {len(df)} matches")
            
            # Filter for complete matches with valid scores
            df = df.dropna(subset=['home_score', 'away_score'])
            df = df[(df['home_score'] >= 0) & (df['away_score'] >= 0)]
            
            # Add outcome column (0=Home, 1=Draw, 2=Away)
            df['outcome'] = 0  # Default to home win
            df.loc[df['home_score'] == df['away_score'], 'outcome'] = 1  # Draw
            df.loc[df['home_score'] < df['away_score'], 'outcome'] = 2   # Away win
            
            # Sample data if requested
            if self.config['sample_size'] and len(df) > self.config['sample_size']:
                df = df.sample(n=self.config['sample_size'], random_state=self.config['random_seed'])
                self.logger.info(f"📊 Sampled {len(df)} matches for evaluation")
            
            # Add required columns for prediction service
            df['league'] = df.get('league_id', 'Unknown').astype(str)
            df['home_team'] = df.get('home_team_id', df.get('home_team', 'Unknown')).astype(str)
            df['away_team'] = df.get('away_team_id', df.get('away_team', 'Unknown')).astype(str)
            
            self.evaluation_data = df
            self.logger.info(f"✅ Evaluation dataset prepared: {len(df)} matches")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error loading evaluation data: {e}")
            raise
    
    def initialize_prediction_service(self):
        """Initialize prediction service"""
        try:
            self.logger.info("🔧 Initializing prediction service...")
            self.prediction_service = ImprovedPredictionService()
            self.logger.info("✅ Prediction service initialized")
        except Exception as e:
            self.logger.error(f"❌ Error initializing prediction service: {e}")
            raise
    
    def generate_predictions(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate predictions for all models
        
        Args:
            df: Evaluation dataset
            
        Returns:
            Dictionary of model_name -> predictions
        """
        try:
            self.logger.info("🎯 Generating predictions for all models...")
            
            predictions = {}
            n_matches = len(df)
            
            # Initialize prediction arrays
            for model in self.config['models_to_evaluate']:
                predictions[model] = np.zeros((n_matches, 3))
            
            # Generate predictions for each match
            for idx, row in df.iterrows():
                if idx % 100 == 0:
                    self.logger.info(f"   Processing match {idx}/{n_matches}")
                
                try:
                    home_team = str(row['home_team'])
                    away_team = str(row['away_team'])
                    league = str(row.get('league', 'Unknown'))
                    
                    # Get hybrid predictions (includes all models)
                    result = self.prediction_service.predict_with_hybrid_1x2(
                        home_team=home_team,
                        away_team=away_team,
                        league=league
                    )
                    
                    # Extract ML predictions
                    if 'ml' in self.config['models_to_evaluate']:
                        ml_1x2 = result.get('prediction_1x2', {})
                        predictions['ml'][idx] = [
                            ml_1x2.get('prob_home_win', 1/3),
                            ml_1x2.get('prob_draw', 1/3),
                            ml_1x2.get('prob_away_win', 1/3)
                        ]
                    
                    # Extract hybrid predictions
                    if 'hybrid' in self.config['models_to_evaluate']:
                        hybrid_1x2 = result.get('prediction_1x2', {}).get('hybrid', {})
                        predictions['hybrid'][idx] = [
                            hybrid_1x2.get('1', 1/3),
                            hybrid_1x2.get('X', 1/3),
                            hybrid_1x2.get('2', 1/3)
                        ]
                    
                    # Extract scoreline predictions
                    if 'scoreline' in self.config['models_to_evaluate']:
                        scoreline_1x2 = result.get('1x2_scoreline', {})
                        if scoreline_1x2:
                            predictions['scoreline'][idx] = [
                                scoreline_1x2.get('p_home', 1/3),
                                scoreline_1x2.get('p_draw', 1/3),
                                scoreline_1x2.get('p_away', 1/3)
                            ]
                        else:
                            predictions['scoreline'][idx] = [1/3, 1/3, 1/3]
                    
                    # Extract Poisson predictions (if available)
                    if 'poisson' in self.config['models_to_evaluate']:
                        # Try to get Poisson from hybrid components
                        hybrid_components = result.get('prediction_1x2', {}).get('hybrid', {}).get('components', {})
                        poisson_comp = hybrid_components.get('poisson')
                        if poisson_comp and isinstance(poisson_comp, dict):
                            predictions['poisson'][idx] = [
                                poisson_comp.get('1', 1/3),
                                poisson_comp.get('X', 1/3),
                                poisson_comp.get('2', 1/3)
                            ]
                        else:
                            predictions['poisson'][idx] = [1/3, 1/3, 1/3]
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Error predicting match {idx}: {e}")
                    # Fill with uniform predictions
                    for model in self.config['models_to_evaluate']:
                        predictions[model][idx] = [1/3, 1/3, 1/3]
            
            # Store predictions
            self.predictions = predictions
            
            self.logger.info("✅ Predictions generated for all models")
            return predictions
            
        except Exception as e:
            self.logger.error(f"❌ Error generating predictions: {e}")
            raise
    
    def evaluate_all_models(self, df: pd.DataFrame, predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models using comprehensive metrics
        
        Args:
            df: Evaluation dataset
            predictions: Model predictions
            
        Returns:
            Dictionary of model_name -> metrics
        """
        try:
            self.logger.info("📊 Evaluating all models...")
            
            y_true = df['outcome'].values
            results = {}
            
            for model_name, y_pred in predictions.items():
                self.logger.info(f"   Evaluating {model_name}...")
                
                # Evaluate model
                metrics = self.evaluator.evaluate_predictions(y_true, y_pred, model_name)
                results[model_name] = metrics
                
                # Log key metrics
                self.logger.info(f"     Accuracy: {metrics['accuracy']:.3f}")
                self.logger.info(f"     Draw Accuracy: {metrics['draw_accuracy']:.3f}")
                self.logger.info(f"     Log Loss: {metrics['log_loss']:.3f}")
                self.logger.info(f"     Calibration Error: {metrics['calibration_error']:.3f}")
            
            # Compare models
            comparison = self.evaluator.compare_models(results)
            results['_comparison'] = comparison
            
            self.logger.info(f"🏆 Best model: {comparison['best_model']}")
            
            # Store results
            self.results = results
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error evaluating models: {e}")
            raise
    
    def generate_plots(self, df: pd.DataFrame, predictions: Dict[str, np.ndarray]) -> Dict[str, str]:
        """
        Generate evaluation plots
        
        Args:
            df: Evaluation dataset
            predictions: Model predictions
            
        Returns:
            Dictionary of plot_name -> file_path
        """
        try:
            if not self.config['save_plots']:
                return {}
            
            self.logger.info("📊 Generating evaluation plots...")
            
            y_true = df['outcome'].values
            plot_dir = Path(self.config['output_dir']) / 'plots'
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_paths = {}
            
            # 1. Calibration curves
            calibration_path = plot_dir / f'calibration_curves_{timestamp}.{self.config["plot_format"]}'
            self.evaluator.plot_calibration_curves(y_true, predictions, str(calibration_path))
            plot_paths['calibration_curves'] = str(calibration_path)
            
            # 2. Distribution comparison
            distribution_path = plot_dir / f'distribution_comparison_{timestamp}.{self.config["plot_format"]}'
            self.evaluator.plot_distribution_comparison(y_true, predictions, str(distribution_path))
            plot_paths['distribution_comparison'] = str(distribution_path)
            
            # 3. Draw accuracy comparison
            draw_accuracy_path = plot_dir / f'draw_accuracy_comparison_{timestamp}.{self.config["plot_format"]}'
            self.evaluator.plot_draw_accuracy_comparison(self.results, str(draw_accuracy_path))
            plot_paths['draw_accuracy_comparison'] = str(draw_accuracy_path)
            
            self.logger.info(f"✅ Generated {len(plot_paths)} plots")
            return plot_paths
            
        except Exception as e:
            self.logger.error(f"❌ Error generating plots: {e}")
            return {}
    
    def save_results(self, results: Dict[str, Any], plot_paths: Dict[str, str]) -> Tuple[str, str]:
        """
        Save evaluation results to JSON and Markdown
        
        Args:
            results: Evaluation results
            plot_paths: Generated plot paths
            
        Returns:
            Tuple of (json_path, markdown_path)
        """
        try:
            self.logger.info("💾 Saving evaluation results...")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path(self.config['output_dir']) / 'reports'
            
            # Save JSON report
            json_path = output_dir / f'hybrid_1x2_report_{timestamp}.json'
            
            # Prepare JSON data
            json_data = {
                'timestamp': timestamp,
                'config': self.config,
                'dataset_info': {
                    'total_matches': len(self.evaluation_data),
                    'home_wins': int(np.sum(self.evaluation_data['outcome'] == 0)),
                    'draws': int(np.sum(self.evaluation_data['outcome'] == 1)),
                    'away_wins': int(np.sum(self.evaluation_data['outcome'] == 2))
                },
                'models_evaluated': list(self.predictions.keys()),
                'results': {k: v for k, v in results.items() if k != '_comparison'},
                'comparison': results.get('_comparison', {}),
                'plots_generated': plot_paths
            }
            
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            # Save Markdown summary
            markdown_path = output_dir / f'hybrid_1x2_summary_{timestamp}.md'
            markdown_content = self._generate_markdown_report(json_data)
            
            with open(markdown_path, 'w') as f:
                f.write(markdown_content)
            
            self.logger.info(f"✅ Results saved:")
            self.logger.info(f"   JSON: {json_path}")
            self.logger.info(f"   Markdown: {markdown_path}")
            
            return str(json_path), str(markdown_path)
            
        except Exception as e:
            self.logger.error(f"❌ Error saving results: {e}")
            return "", ""
    
    def _generate_markdown_report(self, data: Dict[str, Any]) -> str:
        """Generate Markdown evaluation report"""
        try:
            timestamp = data['timestamp']
            results = data['results']
            comparison = data['comparison']
            dataset_info = data['dataset_info']
            
            md = f"""# Hybrid 1X2 Model Evaluation Report

**Generated:** {timestamp}

## Dataset Summary

- **Total Matches:** {dataset_info['total_matches']:,}
- **Home Wins:** {dataset_info['home_wins']:,} ({dataset_info['home_wins']/dataset_info['total_matches']*100:.1f}%)
- **Draws:** {dataset_info['draws']:,} ({dataset_info['draws']/dataset_info['total_matches']*100:.1f}%)
- **Away Wins:** {dataset_info['away_wins']:,} ({dataset_info['away_wins']/dataset_info['total_matches']*100:.1f}%)

## Model Performance Comparison

### Overall Results

| Model | Accuracy | Draw Accuracy | Log Loss | Brier Score | Calibration Error |
|-------|----------|---------------|----------|-------------|-------------------|
"""
            
            # Add model results to table
            for model_name, metrics in results.items():
                md += f"| {model_name.title()} | {metrics['accuracy']:.3f} | {metrics['draw_accuracy']:.3f} | {metrics['log_loss']:.3f} | {metrics['brier_score']:.3f} | {metrics['calibration_error']:.3f} |\n"
            
            md += f"""
### Key Findings

**🏆 Best Overall Model:** {comparison.get('best_model', 'Unknown')}

"""
            
            # Add summary findings
            summary = comparison.get('summary', {})
            for key, value in summary.items():
                md += f"- **{key.replace('_', ' ').title()}:** {value}\n"
            
            md += """
### Detailed Analysis

#### Accuracy Metrics
"""
            
            # Detailed accuracy breakdown
            for model_name, metrics in results.items():
                md += f"""
**{model_name.title()} Model:**
- Overall Accuracy: {metrics['accuracy']:.3f}
- Home Win Accuracy: {metrics['home_accuracy']:.3f}
- Draw Accuracy: {metrics['draw_accuracy']:.3f}
- Away Win Accuracy: {metrics['away_accuracy']:.3f}
"""
            
            md += """
#### Calibration Analysis

Calibration measures how well predicted probabilities match actual outcomes:
"""
            
            for model_name, metrics in results.items():
                md += f"- **{model_name.title()}:** {metrics['calibration_error']:.3f} (lower is better)\n"
            
            md += """
#### Bias Analysis

Prediction bias shows systematic over/under-estimation:
"""
            
            for model_name, metrics in results.items():
                md += f"""
**{model_name.title()} Bias:**
- Home Bias: {metrics.get('home_bias', 0.0):+.3f}
- Draw Bias: {metrics.get('draw_bias', 0.0):+.3f}
- Away Bias: {metrics.get('away_bias', 0.0):+.3f}
"""
            
            # Improvements section
            improvements = comparison.get('improvements', {})
            if improvements:
                md += """
### Model Improvements

Improvements relative to baseline (first model):
"""
                
                for model_name, model_improvements in improvements.items():
                    md += f"""
**{model_name.title()} vs Baseline:**
"""
                    for metric, improvement in model_improvements.items():
                        sign = "+" if improvement > 0 else ""
                        md += f"- {metric.replace('_', ' ').title()}: {sign}{improvement:.1f}%\n"
            
            md += """
## Conclusions and Recommendations

### Model Selection
"""
            
            best_model = comparison.get('best_model', 'Unknown')
            md += f"""
The **{best_model.title()} model** performs best overall based on composite scoring that weighs:
- Overall accuracy (30%)
- Draw detection accuracy (20%)
- Log loss (20%)
- Brier score (15%)
- Calibration error (15%)
"""
            
            md += """
### Key Insights

1. **Draw Detection:** This is typically the most challenging aspect of 1X2 prediction
2. **Calibration:** Well-calibrated models provide more reliable probability estimates
3. **Bias:** Models may systematically favor certain outcomes

### Recommendations

1. **For Production Use:** Deploy the best-performing model based on your specific requirements
2. **For Risk Management:** Consider calibration quality for probability-based decisions
3. **For Draw Betting:** Focus on models with highest draw detection accuracy
4. **For Ensemble:** Combine models with complementary strengths

---

*This report was generated automatically by the Hybrid 1X2 Evaluation Pipeline.*
"""
            
            return md
            
        except Exception as e:
            return f"# Evaluation Report\n\nError generating report: {e}"
    
    def run_complete_evaluation(self) -> Dict[str, str]:
        """
        Run complete evaluation pipeline
        
        Returns:
            Dictionary with output file paths
        """
        try:
            self.logger.info("🚀 Starting complete 1X2 model evaluation...")
            
            # 1. Load evaluation data
            df = self.load_evaluation_data()
            
            # 2. Initialize prediction service
            self.initialize_prediction_service()
            
            # 3. Generate predictions
            predictions = self.generate_predictions(df)
            
            # 4. Evaluate all models
            results = self.evaluate_all_models(df, predictions)
            
            # 5. Generate plots
            plot_paths = self.generate_plots(df, predictions)
            
            # 6. Save results
            json_path, markdown_path = self.save_results(results, plot_paths)
            
            # 7. Summary
            output_files = {
                'json_report': json_path,
                'markdown_summary': markdown_path,
                **plot_paths
            }
            
            self.logger.info("🎉 Evaluation completed successfully!")
            self.logger.info(f"📊 Best model: {results.get('_comparison', {}).get('best_model', 'Unknown')}")
            
            return output_files
            
        except Exception as e:
            self.logger.error(f"❌ Evaluation failed: {e}")
            raise


def main():
    """Main execution"""
    print("🎯 Hybrid 1X2 Model Evaluation Pipeline")
    print("=" * 50)
    
    # Configuration
    config = {
        'sample_size': 500,  # Smaller sample for demo
        'models_to_evaluate': ['ml', 'hybrid', 'scoreline'],
        'save_plots': True
    }
    
    # Initialize and run pipeline
    pipeline = Hybrid1X2EvaluationPipeline(config)
    
    try:
        output_files = pipeline.run_complete_evaluation()
        
        print("\n✅ EVALUATION COMPLETED SUCCESSFULLY")
        print(f"\n📊 Output Files:")
        for file_type, file_path in output_files.items():
            if file_path:
                print(f"   {file_type}: {file_path}")
        
        print(f"\n🏆 Best Model: {pipeline.results.get('_comparison', {}).get('best_model', 'Unknown')}")
        
    except Exception as e:
        print(f"\n❌ EVALUATION FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
