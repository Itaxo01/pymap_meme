"""
World Map Visualization with Top N Cities by Population
Select a country and visualize its N biggest cities on a map.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
import numpy as np
import math
from io import StringIO
from pathlib import Path
import json
from datetime import datetime


# Cache configuration
CACHE_DIR = Path(__file__).parent / "cache"
CITIES_CACHE_FILE = CACHE_DIR / "cities_data.csv"
CACHE_INFO_FILE = CACHE_DIR / "cache_info.json"
CACHE_MAX_AGE_DAYS = 30  # Re-download data after 30 days


def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(exist_ok=True)


def is_cache_valid():
    """Check if the cached data exists and is not too old."""
    if not CITIES_CACHE_FILE.exists() or not CACHE_INFO_FILE.exists():
        return False
    
    try:
        with open(CACHE_INFO_FILE, 'r') as f:
            cache_info = json.load(f)
        
        cached_date = datetime.fromisoformat(cache_info['downloaded_at'])
        age_days = (datetime.now() - cached_date).days
        
        return age_days < CACHE_MAX_AGE_DAYS
    except Exception:
        return False


def save_to_cache(cities_df, source_url):
    """Save cities data to local cache."""
    ensure_cache_dir()
    
    # Save the DataFrame
    cities_df.to_csv(CITIES_CACHE_FILE, index=False)
    
    # Save cache metadata
    cache_info = {
        'downloaded_at': datetime.now().isoformat(),
        'source_url': source_url,
        'num_cities': len(cities_df),
        'num_countries': cities_df['country'].nunique()
    }
    
    with open(CACHE_INFO_FILE, 'w') as f:
        json.dump(cache_info, f, indent=2)
    
    print(f"✓ Cities data cached to: {CITIES_CACHE_FILE}")
    print(f"  ({len(cities_df):,} cities from {cities_df['country'].nunique()} countries)")


def load_from_cache():
    """Load cities data from local cache."""
    if not CITIES_CACHE_FILE.exists():
        return None
    
    try:
        cities_df = pd.read_csv(CITIES_CACHE_FILE)
        
        # Load cache info for display
        if CACHE_INFO_FILE.exists():
            with open(CACHE_INFO_FILE, 'r') as f:
                cache_info = json.load(f)
            cached_date = datetime.fromisoformat(cache_info['downloaded_at'])
            print(f"✓ Loaded cities from cache (downloaded: {cached_date.strftime('%Y-%m-%d')})")
        else:
            print("✓ Loaded cities from cache")
        
        return cities_df
    except Exception as e:
        print(f"⚠ Could not load cache: {e}")
        return None


def clear_cache():
    """Clear the cached data to force re-download."""
    if CITIES_CACHE_FILE.exists():
        CITIES_CACHE_FILE.unlink()
    if CACHE_INFO_FILE.exists():
        CACHE_INFO_FILE.unlink()
    print("✓ Cache cleared. Data will be re-downloaded on next run.")


def get_world_countries():
    """Load world countries geometries from natural earth dataset."""
    world = gpd.read_file(
        "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    )
    return world


def get_cities_data(force_refresh=False):
    """
    Load world cities dataset with population data.
    Uses local cache if available, otherwise downloads from API.
    
    Parameters:
    -----------
    force_refresh : bool
        If True, ignore cache and download fresh data
    """
    # Check cache first (unless force refresh)
    if not force_refresh and is_cache_valid():
        cached_data = load_from_cache()
        if cached_data is not None:
            return cached_data
    
    # Download from API
    geonames_url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/geonames-all-cities-with-a-population-1000/exports/csv?limit=-1&timezone=UTC"
    
    try:
        print("Downloading cities database (this may take a moment)...")
        response = requests.get(geonames_url, timeout=60)
        response.raise_for_status()
        cities_df = pd.read_csv(StringIO(response.text), sep=';')
        
        # Standardize column names
        cities_df = cities_df.rename(columns={
            'name': 'city',
            'cou_name_en': 'country',
            'population': 'population',
            'coordinates': 'coordinates'
        })
        
        # Parse coordinates
        cities_df[['lat', 'lon']] = cities_df['coordinates'].str.split(',', expand=True).astype(float)
        
        # Keep only needed columns
        cities_df = cities_df[['city', 'country', 'population', 'lat', 'lon']].dropna()
        
        # Save to cache for future use
        save_to_cache(cities_df, geonames_url)
        
        return cities_df
        
    except Exception as e:
        cached_data = load_from_cache()
        if cached_data is not None:
            print("Using cached data")
            return cached_data
        print(f"✗ Failed to download cities data: {e}")
        sys.exit(1)



def get_available_countries(cities_df):
    """Get list of unique countries in the dataset."""
    return sorted(cities_df['country'].unique().tolist())


def get_top_cities(cities_df, country, n=10):
    """Get top N cities by population for a given country."""
    country_cities = cities_df[cities_df['country'].str.lower() == country.lower()].copy()
    
    if country_cities.empty:
        return None
    
    return country_cities.nlargest(n, 'population')


def lagrange_interpolation(x_points, y_points):
    """
    Compute Lagrange interpolating polynomial coefficients.
    
    For N points, returns the unique polynomial of degree N-1 that passes
    through all points.
    
    Uses numpy's polyfit which is more numerically stable for higher degrees.
    
    Parameters:
    -----------
    x_points : array-like
        X coordinates of the points
    y_points : array-like
        Y coordinates of the points
    
    Returns:
    --------
    coefficients : numpy array
        Polynomial coefficients from highest to lowest degree
    """
    n = len(x_points)
    x = np.array(x_points, dtype=np.float64)
    y = np.array(y_points, dtype=np.float64)
    
    # Normalize x values to improve numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std == 0:
        x_std = 1
    x_normalized = (x - x_mean) / x_std
    
    # Use numpy's polyfit for better numerical stability
    # Degree is n-1 for n points (Lagrange interpolation)
    degree = n - 1
    
    # Fit polynomial to normalized data
    coeffs_normalized = np.polyfit(x_normalized, y, degree)
    
    # Convert coefficients back to original x scale
    # We need to expand P((x - mean) / std) to get P(x)
    # This is done by computing the polynomial at original scale
    coefficients = _denormalize_polynomial(coeffs_normalized, x_mean, x_std, degree)
    
    return coefficients


def _denormalize_polynomial(coeffs_normalized, x_mean, x_std, degree):
    """
    Convert polynomial coefficients from normalized to original scale.
    
    If P_norm(t) = sum(c_i * t^i) where t = (x - mean) / std,
    we compute P(x) = P_norm((x - mean) / std) in terms of x.
    """
    n = degree + 1
    # We'll compute the coefficients by expanding the polynomial
    # P(x) = sum_{i=0}^{degree} c_i * ((x - mean) / std)^i
    
    # Build coefficient matrix for powers of x
    result = np.zeros(n, dtype=np.float64)
    
    for i, c in enumerate(reversed(coeffs_normalized)):  # i is the power
        # c * ((x - mean) / std)^i
        # = c / std^i * (x - mean)^i
        # = c / std^i * sum_{k=0}^{i} C(i,k) * x^k * (-mean)^(i-k)
        scale = c / (x_std ** i)
        
        for k in range(i + 1):
            # Binomial coefficient C(i, k)
            binom = math.factorial(i) // (math.factorial(k) * math.factorial(i - k))
            contrib = scale * binom * ((-x_mean) ** (i - k))
            result[n - 1 - k] += contrib
    
    return result


def format_coefficient(coef, precision=15):
    """
    Format a coefficient using scientific notation with × 10^x format.
    
    Parameters:
    -----------
    coef : float
        The coefficient to format
    precision : int
        Number of significant digits (default: 15)
    
    Returns:
    --------
    str : Formatted coefficient string
    """
    if coef == 0:
        return "0"
    
    # Get the exponent
    exponent = int(np.floor(np.log10(abs(coef))))
    
    # For small exponents, just show the number directly
    if -4 <= exponent <= 6:
        # Round to precision significant figures
        rounded = round(coef, precision - 1 - exponent)
        # Format without trailing zeros
        if rounded == int(rounded):
            return str(int(rounded))
        else:
            return f"{rounded:.{precision}g}"
    
    # For large/small numbers, use × 10^x notation
    mantissa = coef / (10 ** exponent)
    mantissa = round(mantissa, precision - 1)
    
    if mantissa == int(mantissa):
        mantissa_str = str(int(mantissa))
    else:
        mantissa_str = f"{mantissa:.{precision-1}g}"
    
    return f"{mantissa_str} × 10^{exponent}"


def format_polynomial(coefficients, precision=15):
    """
    Format polynomial coefficients as a human-readable string.
    Uses × 10^x notation instead of e+ notation.
    
    Parameters:
    -----------
    coefficients : array-like
        Polynomial coefficients from highest to lowest degree
    precision : int
        Number of significant digits for coefficients (default: 15)
    
    Returns:
    --------
    str : Formatted polynomial string
    """
    coeffs = np.array(coefficients)
    # Round to specified precision
    coeffs = np.round(coeffs, precision)
    degree = len(coeffs) - 1
    terms = []
    
    for i, coef in enumerate(coeffs):
        power = degree - i
        
        if abs(coef) < 1e-15:  # Skip near-zero coefficients
            continue
        
        # Format coefficient
        coef_str = format_coefficient(abs(coef), precision)
        sign = "-" if coef < 0 else ""
        
        # Build term
        if power == 0:
            term = f"{sign}{coef_str}"
        elif power == 1:
            if abs(coef) == 1:
                term = f"{sign}x"
            else:
                term = f"{sign}{coef_str}x"
        else:
            if abs(coef) == 1:
                term = f"{sign}x^{power}"
            else:
                term = f"{sign}{coef_str}x^{power}"
        
        terms.append((term, coef < 0))
    
    if not terms:
        return "0"
    
    # Join terms with proper signs
    result = terms[0][0]
    for term, is_negative in terms[1:]:
        if is_negative:
            # Term already has minus sign
            result += f" {term}"
        else:
            result += f" + {term}"
    
    return f"P(x) = {result}"


def plot_country_with_lagrange(country_name, num_cities=10, save_path=None):
    """
    Plot a country map with cities and the Lagrange interpolating polynomial.
    
    Uses city longitudes as x-coordinates and latitudes as y-coordinates
    to compute a unique polynomial of degree (N-1) passing through N cities.
    
    Parameters:
    -----------
    country_name : str
        Name of the country to plot
    num_cities : int
        Number of top cities to display (default: 10)
    save_path : str, optional
        Path to save the figure (if None, displays interactively)
    """
    # Load data
    print(f"Loading world map data...")
    world = get_world_countries()
    
    print(f"Loading cities data...")
    cities_df = get_cities_data()
    
    # Find country in world dataset
    country_match = world[world['NAME'].str.lower() == country_name.lower()]
    
    if country_match.empty:
        country_match = world[world['NAME'].str.lower().str.contains(country_name.lower())]
    
    if country_match.empty:
        print(f"\nCountry '{country_name}' not found in map data.")
        return None
    
    country_geom = country_match.iloc[0]
    
    # Get top cities
    top_cities = get_top_cities(cities_df, country_name, num_cities)
    
    if top_cities is None or top_cities.empty:
        print(f"\nNo cities found for '{country_name}' in cities database.")
        return None
    
    # Sort cities by longitude for better polynomial visualization
    top_cities = top_cities.sort_values('lon')
    
    # Extract coordinates
    x_coords = top_cities['lon'].values
    y_coords = top_cities['lat'].values
    
    # Check for duplicate x values (would make polynomial undefined)
    if len(x_coords) != len(set(x_coords)):
        print("⚠ Warning: Some cities have the same longitude.")
        print("  Adding small perturbation to avoid undefined polynomial.")
        # Add tiny perturbation to duplicates
        seen = {}
        for i, x in enumerate(x_coords):
            if x in seen:
                x_coords[i] += 0.0001 * (seen[x] + 1)
                seen[x] += 1
            else:
                seen[x] = 0
    
    # Compute Lagrange polynomial
    print(f"Computing Lagrange polynomial of degree {len(x_coords) - 1}...")
    coefficients = lagrange_interpolation(x_coords, y_coords)
    # Round coefficients to 15 digits
    coefficients = np.round(coefficients, 15)
    polynomial_str = format_polynomial(coefficients)
    
    # Create x values for plotting the polynomial curve
    x_min, x_max = x_coords.min(), x_coords.max()
    x_range = x_max - x_min
    x_plot = np.linspace(x_min - 0.05 * x_range, x_max + 0.05 * x_range, 500)
    y_plot = np.polyval(coefficients, x_plot)
    
    # Get country bounds for limiting the plot view
    country_bounds = country_match.total_bounds  # [minx, miny, maxx, maxy]
    y_min_country, y_max_country = country_bounds[1], country_bounds[3]
    y_range_country = y_max_country - y_min_country
    
    # Calculate reasonable y-limits based on country and city positions
    y_min_cities, y_max_cities = y_coords.min(), y_coords.max()
    y_padding = max(y_range_country * 0.5, (y_max_cities - y_min_cities) * 0.5)
    y_limit_min = min(y_min_country, y_min_cities) - y_padding
    y_limit_max = max(y_max_country, y_max_cities) + y_padding
    
    # Clip polynomial values to reasonable range for display
    y_plot_clipped = np.clip(y_plot, y_limit_min - y_padding, y_limit_max + y_padding)
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Plot the country
    country_match.plot(ax=ax, color='lightgreen', edgecolor='darkgreen', linewidth=2)
    
    # Plot the Lagrange polynomial curve (clipped for display)
    ax.plot(x_plot, y_plot_clipped, 'b-', linewidth=2, label=f'Lagrange Polynomial (degree {len(x_coords)-1})', zorder=4)
    
    # Set axis limits to keep the map visible
    ax.set_ylim(y_limit_min, y_limit_max)
    
    # Plot cities as dots
    ax.scatter(
        x_coords, 
        y_coords,
        s=100,
        c='red',
        alpha=0.9,
        edgecolors='darkred',
        linewidth=1.5,
        zorder=5,
        label='Cities'
    )
    
    # Add city labels
    for _, row in top_cities.iterrows():
        ax.annotate(
            row['city'],
            xy=(row['lon'], row['lat']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            color='darkblue',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )
    
    # Set title
    ax.set_title(
        f"Lagrange Interpolating Polynomial through Top {len(top_cities)} Cities in {country_geom['NAME']}",
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left')
    
    # Add polynomial formula in a text box
    # For long formulas, show simplified version
    if len(polynomial_str) > 100:
        display_formula = f"P(x) = polynomial of degree {len(x_coords)-1}\n(coefficients shown below)"
        coef_text = "Coefficients (highest to lowest degree):\n"
        for i, c in enumerate(coefficients):
            power = len(coefficients) - 1 - i
            coef_text += f"  x^{power}: {format_coefficient(c)}\n"
        formula_text = display_formula + "\n\n" + coef_text
    else:
        formula_text = polynomial_str
    
    # Add city info (sorted by population for display)
    cities_by_pop = top_cities.sort_values('population', ascending=False)
    info_text = "Cities (by population):\n"
    for i, (_, row) in enumerate(cities_by_pop.iterrows(), 1):
        pop_formatted = f"{row['population']:,.0f}"
        info_text += f"{i}. {row['city']}: {pop_formatted}\n"
    
    # Text box with polynomial formula (bottom)
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
    ax.text(
        0.02, -0.12, formula_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment='top',
        bbox=props,
        family='monospace'
    )
    
    # Text box with city list (right side)
    props2 = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(
        1.02, 0.98, info_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=props2,
        family='monospace'
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for formula
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nMap saved to: {save_path}")
    
    plt.show()
    
    # Print polynomial to console as well
    print("\n" + "="*60)
    print("LAGRANGE INTERPOLATING POLYNOMIAL")
    print("="*60)
    print(f"Degree: {len(x_coords) - 1}")
    print(f"Points: {len(x_coords)} cities")
    print("\nPolynomial coefficients (highest to lowest degree):")
    for i, c in enumerate(coefficients):
        power = len(coefficients) - 1 - i
        sign = "+" if c >= 0 else ""
        print(f"  x^{power}: {sign}{format_coefficient(c)}")
    print("="*60)
    
    return fig, ax, coefficients


def plot_country_with_cities(country_name, num_cities=10, save_path=None):
    """
    Plot a country map with its top N cities highlighted by population.
    
    Parameters:
    -----------
    country_name : str
        Name of the country to plot
    num_cities : int
        Number of top cities to display (default: 10)
    save_path : str, optional
        Path to save the figure (if None, displays interactively)
    """
    # Load data
    print(f"Loading world map data...")
    world = get_world_countries()
    
    print(f"Loading cities data...")
    cities_df = get_cities_data()
    
    # Find country in world dataset (handle naming differences)
    country_match = world[world['NAME'].str.lower() == country_name.lower()]
    
    if country_match.empty:
        # Try partial match
        country_match = world[world['NAME'].str.lower().str.contains(country_name.lower())]
    
    if country_match.empty:
        print(f"\nCountry '{country_name}' not found in map data.")
        print("Available countries:")
        for name in sorted(world['NAME'].unique()):
            print(f"  - {name}")
        return None
    
    country_geom = country_match.iloc[0]
    
    # Get top cities
    top_cities = get_top_cities(cities_df, country_name, num_cities)
    
    if top_cities is None or top_cities.empty:
        print(f"\nNo cities found for '{country_name}' in cities database.")
        print("\nAvailable countries in cities database:")
        available = get_available_countries(cities_df)
        for name in available[:50]:  # Show first 50
            print(f"  - {name}")
        if len(available) > 50:
            print(f"  ... and {len(available) - 50} more")
        return None
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Plot the country
    country_match.plot(ax=ax, color='lightgreen', edgecolor='darkgreen', linewidth=2)
    
    # Plot cities as simple dots
    ax.scatter(
        top_cities['lon'], 
        top_cities['lat'],
        s=80,
        c='red',
        alpha=0.8,
        edgecolors='darkred',
        linewidth=1.5,
        zorder=5
    )
    
    # Add city labels
    for idx, row in top_cities.iterrows():
        ax.annotate(
            row['city'],
            xy=(row['lon'], row['lat']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            color='darkblue',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )
    
    # Set title and labels
    ax.set_title(
        f"Top {len(top_cities)} Cities in {country_geom['NAME']} by Population",
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add legend for city info
    info_text = "Cities (by population):\n"
    for i, (_, row) in enumerate(top_cities.iterrows(), 1):
        pop_formatted = f"{row['population']:,.0f}"
        info_text += f"{i}. {row['city']}: {pop_formatted}\n"
    
    # Add text box with city list
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(
        1.02, 0.98, info_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=props,
        family='monospace'
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nMap saved to: {save_path}")
    
    plt.show()
    
    return fig, ax


def interactive_mode(country="default", num_cities=-1):

    if country == "default":
      country = input("\nEnter country name (or 'refresh' to update cache): ").strip()
    
	 # Handle cache refresh command
    if country.lower() == 'refresh':
        print("\nRefreshing cities database...")
        cities_df = get_cities_data(force_refresh=True)
        country = input("\nEnter country name: ").strip()
    
    if(num_cities == -1):
     try:
         num_cities = int(input("Enter number of cities to display (default 10): ").strip() or "10")
     except ValueError:
         num_cities = 10
    
    # Ask for visualization mode
    
    suffix = "_lagrange"
    save_path = f"{country.replace(' ', '_').lower()}_top_{num_cities}_cities{suffix}.png"
    
    print("\n" + "-" * 60)
    print(f"Generating map for {country} with top {num_cities} cities...")
    print("-" * 60 + "\n")
    
    plot_country_with_lagrange(country, num_cities, save_path)


if __name__ == "__main__":
    import sys
    
    # Handle command line arguments for cache management
    if len(sys.argv) > 1:
        if sys.argv[1] == '--clear-cache':
            clear_cache()
            sys.exit(0)
        elif sys.argv[1] == '--refresh':
            print("Refreshing cities database...")
            get_cities_data(force_refresh=True)
            sys.exit(0)
        elif sys.argv[1] == '--help':
            print("Usage: python city_map.py [OPTIONS]")
            print("\nOptions:")
            print("  --clear-cache  Clear cached cities data")
            print("  --refresh      Force re-download cities data")
            print("  --help         Show this help message")
            print("\nWithout options, runs in interactive mode.")
            sys.exit(0)
        elif len(sys.argv) == 3:
            try:
                interactive_mode(sys.argv[1], int(sys.argv[2]))
                
            except any:
                print("Error")
                exit()
    else:
        interactive_mode()
            
          
            
	
    
