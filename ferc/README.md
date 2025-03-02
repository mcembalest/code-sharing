# FERC EQR Report Viewer Downloader

A simple and reliable tool to download transaction data from the FERC EQR Report Viewer website.

## Features

- Downloads transaction data from FERC EQR Report Viewer
- Filter by date range, seller, and balancing authority
- Downloads data in CSV format
- Supports downloading data for all sellers or a specific seller
- Detailed logging for troubleshooting

## Requirements

- Python 3.8 or higher
- Selenium
- Chrome browser installed
- ChromeDriver (automatically installed with Selenium)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ferc.git
   cd ferc
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   or using `uv`:
   ```
   uv pip install .
   ```

## Usage

Run the script in interactive mode:

```
python main.py
```

You will be prompted to enter:
- Start date (MM/DD/YYYY format)
- End date (MM/DD/YYYY format) 
- Seller name (or leave blank for all sellers)
- Balancing authority (defaults to "CISO" if left blank)
- Download directory (defaults to current directory if left blank)
- Headless mode (y/n, run without visible browser)

Alternatively, run with the `--default` flag to use default values for all parameters:

```
python main.py --default
```

This will use:
- Start date: First day of the current quarter
- End date: Current date
- Seller: all
- Authority: CISO
- Download directory: Current directory
- Headless mode: off (browser will be visible)

## Example

```
Enter start date (MM/DD/YYYY): 10/01/2024
Enter end date (MM/DD/YYYY): 11/01/2024
Enter seller name (or just hit enter for all sellers): 3PR Trading, Inc.
Enter authority (default is CISO): 
Enter download directory (or press Enter for current directory): ./data
Run in headless mode? (y/n, default: n): n
```

## How It Works

This script uses Selenium WebDriver to automate interactions with the FERC EQR Report Viewer website:

1. It opens a Chrome browser (visible or headless)
2. Navigates to the Filing Inquiries tab
3. Sets the appropriate form values (Report Type, BA and HUB, dates, etc.)
4. Selects the specified seller (or iterates through all sellers)
5. Submits the form to download the CSV data
6. Monitors the download directory to confirm successful downloads

## Troubleshooting

If the script encounters issues:

1. Make sure Chrome is installed and up to date
2. Try running without headless mode to see what's happening
3. Check that your date range is valid (within available report periods)
4. Verify that the seller name exactly matches one in the dropdown
5. Try with a specific seller instead of "all"
6. Increase the WebDriverWait timeout if the website is slow

## License

[Your License Here]