# -*- coding: utf-8 -*-
"""
Distribution Fitting Tool: Input Analysis with Desk-DistFit
============================================================
`desk-distfit` is the official DESK input-analysis CLI for 
statistically fitting probability distributions to empirical data. 

DESK adopts a verb-oriented command-line interface, where simulation 
tasks are expressed as structured actions (`desk-distfit`), 
ensuring consistency, reproducibility, and ease of learning across 
the framework. Fit probability distributions to empirical data.

Author: João Flávio F. Almeida (PPGEP-UFMG) <joao.flavio@dep.ufmg.br>
Course: EPD899: Simulating Logistics Systems
"""

import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import logging
import argparse
import sys


import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configure matplotlib
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['font.size'] = 11


@dataclass
class DistributionResult:
    """Data class to store distribution fitting results."""
    name: str
    statistic: float
    p_value: float
    parameters: Tuple
    sse: float = None
    is_significant: bool = None


class DistributionFitter:
    """
    A class for fitting probability distributions to empirical data.
    """
    
    # Default distributions to test
    DEFAULT_DISTRIBUTIONS = [
        'uniform', 'triang', 'expon', 'norm', 'lognorm',
        'beta', 'gamma', 'weibull_min', 'weibull_max'
    ]
    
    # Python random module mapping
    PYTHON_RANDOM_MAP = {
        'uniform': ('uniform', lambda loc, scale: f"random.uniform({loc:.3f}, {loc + scale:.3f})"),
        'triang': ('triangular', lambda params: f"random.triangular({params['low']:.3f}, {params['high']:.3f}, {params['mode']:.3f})"),
        'expon': ('expovariate', lambda loc, scale: f"random.expovariate({1/scale:.3f})"),
        'norm': ('gauss', lambda loc, scale: f"random.gauss({loc:.3f}, {scale:.3f})"),
        'lognorm': ('lognormvariate', lambda loc, scale: f"random.lognormvariate({loc:.3f}, {scale:.3f})"),
        'beta': ('betavariate', lambda params: f"random.betavariate({params['a']:.3f}, {params['b']:.3f})"),
        'gamma': ('gammavariate', lambda params: f"random.gammavariate({params['a']:.3f}, {1/params['scale']:.3f})"),
        'weibull_min': ('weibullvariate', lambda loc, scale: f"random.weibullvariate({scale:.3f}, {loc:.3f})"),
        'weibull_max': ('weibullvariate', lambda loc, scale: f"random.weibullvariate({scale:.3f}, {loc:.3f})")
    }
    
    def __init__(self, alpha: float = 0.05, bins: int = 50):
        """
        Initialize the distribution fitter.
        
        Args:
            alpha: Significance level for statistical tests
            bins: Number of bins for histogram
        """
        self.alpha = alpha
        self.bins = bins
        self.results: List[DistributionResult] = []
        self.data: Optional[pd.Series] = None
        
    def load_data(self, filepath: Union[str, Path]) -> pd.Series:
        """
        Load data from a text file.
        
        Args:
            filepath: Path to the data file
            
        Returns:
            Pandas Series containing the data
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"File {filepath} not found")
                
            with open(filepath, 'r', encoding='utf-8') as file:
                data_list = [float(line.strip()) for line in file if line.strip()]
                
            self.data = pd.Series(data_list, name='data')
            logger.info(f"Loaded {len(self.data)} data points from {filepath}")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
            
    def set_data(self, data: Union[List, np.ndarray, pd.Series]) -> pd.Series:
        """
        Set data directly from array-like object.
        
        Args:
            data: Array-like data
            
        Returns:
            Pandas Series containing the data
        """
        self.data = pd.Series(data, name='data')
        logger.info(f"Set data with {len(self.data)} points")
        return self.data
        
    @staticmethod
    def get_parameter_names(distribution: Union[str, st.rv_continuous]) -> List[str]:
        """
        Get parameter names for a given distribution.
        
        Args:
            distribution: Distribution name or scipy.stats distribution object
            
        Returns:
            List of parameter names
        """
        if isinstance(distribution, str):
            distribution = getattr(st, distribution)
            
        parameters = []
        if distribution.shapes:
            parameters = [name.strip() for name in distribution.shapes.split(',')]
            
        # Add location and scale parameters
        if hasattr(distribution, 'name'):
            if distribution.name in st._discrete_distns._distn_names:
                parameters += ['loc']
            elif distribution.name in st._continuous_distns._distn_names:
                parameters += ['loc', 'scale']
                
        return parameters
        
    def fit_distributions(self, distributions: Optional[List[str]] = None) -> List[DistributionResult]:
        """
        Fit multiple distributions to the data and perform goodness-of-fit tests.
        
        Args:
            distributions: List of distribution names to test
            
        Returns:
            List of DistributionResult objects sorted by p-value (descending)
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() or set_data() first.")
            
        distributions = distributions or self.DEFAULT_DISTRIBUTIONS
        self.results = []
        
        # Get histogram for SSE calculation
        y, x = np.histogram(self.data, bins=self.bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0
        
        logger.info(f"Fitting {len(distributions)} distributions...")
        print(f"{'Item':<5}{'Distribution':<15}{'Statistic':<12}{'P-value':<12}{'Significant'}")
        print("-" * 60)
        
        for i, dist_name in enumerate(distributions):
            try:
                dist = getattr(st, dist_name)
                
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    
                    # Fit distribution parameters
                    params = dist.fit(self.data)
                    
                    # Perform Kolmogorov-Smirnov test
                    ks_stat, p_value = st.kstest(self.data, dist_name, params)
                    
                    # Calculate SSE for ranking
                    arg = params[:-2] if len(params) > 2 else []
                    loc, scale = params[-2], params[-1]
                    pdf = dist.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum((y - pdf) ** 2)
                    
                    is_significant = p_value >= self.alpha
                    result = DistributionResult(
                        name=dist_name,
                        statistic=ks_stat,
                        p_value=p_value,
                        parameters=params,
                        sse=sse,
                        is_significant=is_significant
                    )
                    
                    self.results.append(result)
                    
                    # Print result
                    sig_mark = " (*)" if is_significant else ""
                    print(f"{i+1:<5}{dist_name:<15}{ks_stat:<12.4f}{p_value:<12.4f}{sig_mark}")
                    
            except Exception as e:
                logger.warning(f"Failed to fit {dist_name}: {e}")
                
        # Sort by p-value (descending)
        self.results.sort(key=lambda x: x.p_value, reverse=True)
        
        logger.info(f"Successfully fitted {len(self.results)} distributions")
        return self.results
        
    def get_best_fit(self) -> Optional[DistributionResult]:
        """Get the best fitting distribution (highest p-value)."""
        return self.results[0] if self.results else None
        
    def print_parameters(self) -> None:
        """Print detailed parameter information for all fitted distributions."""
        if not self.results:
            logger.warning("No results available. Run fit_distributions() first.")
            return
            
        print("\nDistribution Parameters:")
        print("=" * 70)
        
        for result in self.results:
            param_names = self.get_parameter_names(result.name)
            print(f"\n{result.name}: {param_names} (p-value = {result.p_value:.4f})")
            
            for j, (name, value) in enumerate(zip(param_names, result.parameters)):
                print(f"  {j+1}: {name} = {value:.4f}")
                
    def get_python_random_code(self, result: DistributionResult) -> str:
        """
        Generate Python random module code for the given distribution.
        
        Args:
            result: DistributionResult object
            
        Returns:
            Python code string for generating random numbers
        """
        dist_name = result.name
        params = result.parameters
        param_names = self.get_parameter_names(dist_name)
        
        if dist_name not in self.PYTHON_RANDOM_MAP:
            return f"# No Python random mapping available for {dist_name}"
            
        param_dict = dict(zip(param_names, params))
        loc = param_dict.get('loc', 0)
        scale = param_dict.get('scale', 1)
        
        try:
            if dist_name == 'uniform':
                return f"random.uniform({loc:.3f}, {loc + scale:.3f})"
                
            elif dist_name == 'triang':
                c = param_dict['c']
                low, high, mode = loc, loc + scale, loc + c * scale
                return f"random.triangular({low:.3f}, {high:.3f}, {mode:.3f})"
                
            elif dist_name == 'expon':
                return f"random.expovariate({1/scale:.3f})"
                
            elif dist_name == 'norm':
                return f"random.gauss({loc:.3f}, {scale:.3f})"
                
            elif dist_name == 'lognorm':
                return f"random.lognormvariate({loc:.3f}, {scale:.3f})"
                
            elif dist_name == 'beta':
                a, b = param_dict['a'], param_dict['b']
                return f"random.betavariate({a:.3f}, {b:.3f})"
                
            elif dist_name == 'gamma':
                a = param_dict['a']
                return f"random.gammavariate({a:.3f}, {1/scale:.3f})"
                
            elif dist_name in ['weibull_min', 'weibull_max']:
                return f"random.weibullvariate({scale:.3f}, {loc:.3f})"
                
        except KeyError as e:
            logger.error(f"Missing parameter {e} for distribution {dist_name}")
            
        return f"# Error generating code for {dist_name}"
        
    def plot_results(self, show_all: bool = False, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot the fitted distributions against the data.
        
        Args:
            show_all: If True, plot all fitted distributions; if False, only the best fit
            figsize: Figure size tuple
        """
        if not self.results or self.data is None:
            logger.warning("No results or data available for plotting")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot histogram of original data
        self.data.hist(bins=self.bins, density=True, alpha=0.7, ax=ax1, 
                      color='skyblue', edgecolor='black')
        ax1.set_title('Data Histogram \n\n Python code for best fit ->')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.grid(True, alpha=0.3)
        
        # Plot best fit or all distributions
        results_to_plot = self.results if show_all else [self.results[0]]
        
        for i, result in enumerate(results_to_plot[:5]):  # Limit to 5 for readability
            try:
                dist = getattr(st, result.name)
                arg = result.parameters[:-2] if len(result.parameters) > 2 else []
                loc, scale = result.parameters[-2], result.parameters[-1]
                
                # Generate PDF
                x_min, x_max = self.data.min(), self.data.max()
                x_range = x_max - x_min
                x = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, 1000)
                y = dist.pdf(x, loc=loc, scale=scale, *arg)
                
                label = f"{result.name} (p={result.p_value:.3f})"
                ax2.plot(x, y, linewidth=2, label=label)
                
            except Exception as e:
                logger.warning(f"Error plotting {result.name}: {e}")
                
        # Add data histogram to comparison plot
        self.data.hist(bins=self.bins, density=True, alpha=0.5, ax=ax2,
                      color='lightgray', label='Data')
        
        best_fit = self.get_best_fit()
        ax2.set_title('Distribution Fit Comparison: \n\n' + self.get_python_random_code(best_fit))
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report."""
        if not self.results:
            return "No results available. Run fit_distributions() first."
            
        best_fit = self.get_best_fit()
        param_names = self.get_parameter_names(best_fit.name)
        param_str = ', '.join([f'{k}={v:.3f}' for k, v in zip(param_names, best_fit.parameters)])
        
        report = f"""
Distribution Fitting Summary Report
{'='*50}

Data Statistics:
- Sample size: {len(self.data)}
- Mean: {self.data.mean():.4f}
- Std Dev: {self.data.std():.4f}
- Min: {self.data.min():.4f}
- Max: {self.data.max():.4f}

Best Fitting Distribution: {best_fit.name}
- Parameters: {param_str}
- P-value: {best_fit.p_value:.4f}
- Significant at α={self.alpha}: {'Yes' if best_fit.is_significant else 'No'}

Python Random Code:
{self.get_python_random_code(best_fit)}

Top 3 Distributions by P-value:
"""
        
        for i, result in enumerate(self.results[:3], 1):
            sig = "Yes" if result.is_significant else "No"
            report += f"{i}. {result.name}: p-value = {result.p_value:.4f} (Significant: {sig})\n"
            
        return report


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="DESK – Distribution Fitting Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  desk-distfit -d input_data/foo.txt                                    # Basic usage
  desk-distfit -d input_data/foo.txt -a 0.01                            # Custom significance level
  desk-distfit -d input_data/foo.txt -b 100                             # Custom bins
  desk-distfit -d input_data/foo.txt --no-plot                          # Skip plotting
  desk-distfit -d input_data/foo.txt --distributions norm expon gamma   # Test specific distributions
  desk-distfit -d input_data/foo.txt -o results.txt --format json       # Save results to file
        """
    )
    
    # Required arguments
    parser.add_argument(
        '-d', '--data',
        type=str,
        required=True,
        help='Path to the data file (required)'
    )
    
    # Optional arguments
    parser.add_argument(
        '-a', '--alpha',
        type=float,
        default=0.05,
        help='Significance level for statistical tests (default: 0.05)'
    )
    
    parser.add_argument(
        '-b', '--bins',
        type=int,
        default=50,
        help='Number of bins for histogram (default: 50)'
    )
    
    parser.add_argument(
        '--distributions',
        nargs='+',
        help='List of distributions to test (default: all supported)',
        choices=['uniform', 'triang', 'expon', 'norm', 'lognorm', 
                'beta', 'gamma', 'weibull_min', 'weibull_max'],
        metavar='DIST'
    )
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip generating plots'
    )
    
    parser.add_argument(
        '--show-all',
        action='store_true',
        help='Show all fitted distributions in plot (default: only best fit)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file to save results (optional)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--format',
        choices=['table', 'json', 'csv'],
        default='table',
        help='Output format for results (default: table)'
    )
    
    return parser


def save_results_to_file(fitter: DistributionFitter, filepath: str, format_type: str = 'table') -> None:
    """
    Save results to a file in the specified format.
    
    Args:
        fitter: DistributionFitter instance with results
        filepath: Path to save the results
        format_type: Format type ('table', 'json', 'csv')
    """
    try:
        if format_type == 'table':
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(fitter.generate_summary_report())
                f.write("\n\nDetailed Results:\n")
                f.write("-" * 80 + "\n")
                
                for i, result in enumerate(fitter.results, 1):
                    param_names = fitter.get_parameter_names(result.name)
                    param_str = ', '.join([f'{k}={v:.4f}' for k, v in zip(param_names, result.parameters)])
                    f.write(f"{i}. {result.name}\n")
                    f.write(f"   Parameters: {param_str}\n")
                    f.write(f"   P-value: {result.p_value:.6f}\n")
                    f.write(f"   Statistic: {result.statistic:.6f}\n")
                    f.write(f"   Python code: {fitter.get_python_random_code(result)}\n\n")
                    
        elif format_type == 'csv':
            import csv
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Distribution', 'P_value', 'Statistic', 'Significant', 'Python_Code'])
                
                for result in fitter.results:
                    writer.writerow([
                        result.name,
                        f"{result.p_value:.6f}",
                        f"{result.statistic:.6f}",
                        "Yes" if result.is_significant else "No",
                        fitter.get_python_random_code(result)
                    ])
                    
        elif format_type == 'json':
            import json
            results_dict = {
                'summary': {
                    'sample_size': len(fitter.data),
                    'best_distribution': fitter.results[0].name if fitter.results else None,
                    'alpha': fitter.alpha
                },
                'results': []
            }
            
            for result in fitter.results:
                param_names = fitter.get_parameter_names(result.name)
                results_dict['results'].append({
                    'distribution': result.name,
                    'p_value': result.p_value,
                    'statistic': result.statistic,
                    'parameters': dict(zip(param_names, result.parameters)),
                    'significant': result.is_significant,
                    'python_code': fitter.get_python_random_code(result)
                })
                
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2, ensure_ascii=False)
                
        logger.info(f"Results saved to {filepath} in {format_type} format")
        
    except Exception as e:
        logger.error(f"Error saving results to file: {e}")


def run_cli(args: argparse.Namespace) -> int:
    """
    Run the distribution fitting with command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Configure logging level
        log_level = logging.DEBUG if args.verbose else logging.INFO
        logging.getLogger().setLevel(log_level)
        
        # Create fitter instance
        fitter = DistributionFitter(alpha=args.alpha, bins=args.bins)
        
        # Load data        
        # Resolve data path exactly like desk-sim
        data_path = resolve_data_path(args.data)

        logger.info(f"Loading data from {data_path}")
        fitter.load_data(data_path)
        
        # Display data statistics
        print(f"\nData Statistics:")
        print(f"Sample size: {len(fitter.data)}")
        print(f"Mean: {fitter.data.mean():.4f}")
        print(f"Std Dev: {fitter.data.std():.4f}")
        print(f"Min: {fitter.data.min():.4f}")
        print(f"Max: {fitter.data.max():.4f}")
        print()
        
        # Fit distributions
        distributions = args.distributions if args.distributions else None
        results = fitter.fit_distributions(distributions)
        
        if not results:
            logger.error("No distributions could be fitted to the data")
            return 1
        
        # Print parameter details
        fitter.print_parameters()
        
        # Generate and print summary report
        print(fitter.generate_summary_report())
        
        # Save results to file if requested
        if args.output:
            save_results_to_file(fitter, args.output, args.format)
        
        # Generate plots if requested
        if not args.no_plot:
            try:
                fitter.plot_results(show_all=args.show_all)
            except Exception as e:
                logger.warning(f"Could not generate plots: {e}")
                logger.info("Continuing without plots...")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid data: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

def resolve_data_path(data_arg: str) -> Path:
    """
    Resolve data file path the same way desk-sim resolves models.
    """
    path = Path(data_arg).expanduser()

    # Absolute path
    if path.is_absolute() and path.exists():
        return path.resolve()

    # Relative to current working directory
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path.resolve()

    raise FileNotFoundError(
        f"Data file not found: {data_arg}\n"
        f"Tried:\n"
        f"  - {cwd_path}\n"
        f"  - {path.resolve()}"
    )

def main():
    """Main entry point for both CLI and direct usage."""
    # Check if script is run with command line arguments
    if len(sys.argv) > 1:
        # Parse command line arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Run CLI version
        exit_code = run_cli(args)
        sys.exit(exit_code)
    else:
        # Run example/demo version (original main functionality)
        logger.info("Running in demo mode (no command line arguments provided)")
        logger.info("Please provide a data file using: desk-distfit -d <filename>")
        logger.info("Use 'desk-distfit -h' for help")
        
        print("\nUsage: desk-distfit -d <data_file>")
        print("Example: desk-distfit -d entrada1.txt")
        print("\nFor more options, use: desk-distfit -h")


if __name__ == "__main__":
    main()