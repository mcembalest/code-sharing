import os
import time
from datetime import datetime
import sys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException


def setup_driver(download_dir, headless=False):
    """Set up and configure the Chrome WebDriver with appropriate options"""
    chrome_options = Options()
    
    # Set download preferences
    prefs = {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    chrome_options.add_experimental_option("prefs", prefs)
    
    # Set additional options
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-extensions")
    
    # Headless mode if requested
    if headless:
        chrome_options.add_argument("--headless=new")
    
    # Create Chrome WebDriver
    try:
        # First try with ChromeDriverManager
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
    except WebDriverException:
        # Fallback to default Chrome service
        print("ChromeDriverManager failed, trying default Chrome driver...")
        driver = webdriver.Chrome(options=chrome_options)
    
    return driver


class FercDownloader:
    """
    A simple tool to download FERC EQR transaction data using Selenium.
    """
    
    def __init__(self, download_dir=None, headless=False):
        """Initialize the FERC downloader with optional download directory"""
        if download_dir is None:
            download_dir = os.getcwd()
        self.download_dir = os.path.abspath(download_dir)
        
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)
            
        print(f"Files will be saved to: {self.download_dir}")
        
        # Set up the Chrome driver
        try:
            self.driver = setup_driver(self.download_dir, headless)
            self.driver.maximize_window()
            
            # Create a WebDriverWait with 30 second timeout
            self.wait = WebDriverWait(self.driver, 30)
        except Exception as e:
            print(f"Error setting up Chrome driver: {str(e)}")
            raise
    
    def determine_report_period(self, start_date, end_date):
        """Determine the appropriate report period based on input dates"""
        start_dt = datetime.strptime(start_date, "%m/%d/%Y")
        end_dt = datetime.strptime(end_date, "%m/%d/%Y") 
        
        # Find the midpoint of the date range
        mid_point = start_dt + (end_dt - start_dt) / 2
        year = mid_point.year
        month = mid_point.month
        
        # Determine quarter based on month
        if 1 <= month <= 3:
            return f"Q1, Jan-Mar {year}"
        elif 4 <= month <= 6:
            return f"Q2, Apr-Jun {year}"
        elif 7 <= month <= 9:
            return f"Q3, Jul-Sep {year}"
        else:
            return f"Q4, Oct-Dec {year}"
    
    def wait_for_element_to_be_active(self, locator, timeout=10):
        """Wait for an element to be present, visible, and enabled"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                element = self.driver.find_element(*locator)
                if element.is_displayed() and element.is_enabled() and not "aspNetDisabled" in element.get_attribute("class"):
                    return element
            except:
                pass
            time.sleep(0.5)
        
        # If we get here, the element wasn't found or wasn't active
        return None
    
    def wait_for_postback(self, timeout=10):
        """Wait for ASP.NET postback to complete"""
        self.driver.execute_script("window.wait_for_postback_complete = false;")
        self.driver.execute_script("""
            if (typeof Sys !== 'undefined' && Sys.WebForms && Sys.WebForms.PageRequestManager) {
                var prm = Sys.WebForms.PageRequestManager.getInstance();
                if (prm) {
                    prm.add_endRequest(function() {
                        window.wait_for_postback_complete = true;
                    });
                }
            } else {
                window.wait_for_postback_complete = true;
            }
        """)
        
        # Wait for the postback to complete
        start_time = time.time()
        while time.time() - start_time < timeout:
            completed = self.driver.execute_script("return window.wait_for_postback_complete;")
            if completed:
                return True
            time.sleep(0.5)
        
        return False
    
    def download_transaction_data(self, start_date, end_date, seller="all", authority="CISO"):
        """
        Download FERC transaction data based on input parameters.
        
        Args:
            start_date (str): Start date in MM/DD/YYYY format
            end_date (str): End date in MM/DD/YYYY format
            seller (str): Specific seller or "all" to iterate through all sellers
            authority (str): Balancing authority (default: "CISO")
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Step 1: Load the FERC EQR Report Viewer page
            print("Opening FERC EQR Report Viewer...")
            self.driver.get("https://eqrreportviewer.ferc.gov/")
            time.sleep(5)  # Give page time to fully load
            print("Page loaded")
            
            # Step 2: Wait for page to load and take a screenshot for debugging
            debug_screenshot = os.path.join(self.download_dir, "initial_page.png")
            self.driver.save_screenshot(debug_screenshot)
            print(f"Saved initial page screenshot to {debug_screenshot}")
            
            # Step 3: First, click the Reports tab (parent tab)
            try:
                # Using the ID from the HTML structure
                reports_tab = self.wait.until(EC.element_to_be_clickable(
                    (By.ID, "__tab_TabContainerReportViewer_TabPanelReporting")))
                reports_tab.click()
                print("Clicked on Reports tab")
                time.sleep(2)
                
                # Now click the Filing Inquiries tab (child tab)
                filing_tab = self.wait.until(EC.element_to_be_clickable(
                    (By.ID, "__tab_TabContainerReportViewer_TabPanelReporting_TabContainerReports_TabPanelFilingInquiries")))
                filing_tab.click()
                print("Clicked on Filing Inquiries tab")
                
            except Exception as tab_error:
                print(f"Error finding and clicking tabs: {str(tab_error)}")
                
                # Try alternative approaches
                try:
                    # Try finding by link text
                    reports_tabs = self.driver.find_elements(By.XPATH, "//a[contains(., 'Reports')]")
                    if reports_tabs:
                        for tab in reports_tabs:
                            print(f"Found Reports tab: {tab.get_attribute('id')}")
                            tab.click()
                            time.sleep(2)
                            break
                    
                    filing_tabs = self.driver.find_elements(By.XPATH, "//a[contains(., 'Filing Inquiries')]")
                    if filing_tabs:
                        for tab in filing_tabs:
                            print(f"Found Filing Inquiries tab: {tab.get_attribute('id')}")
                            tab.click()
                            time.sleep(2)
                            break
                    
                except Exception as alt_error:
                    print(f"Alternative approach also failed: {str(alt_error)}")
                
                # Try using JavaScript to click the tabs
                try:
                    print("Trying JavaScript approach...")
                    # First, locate the Reports tab by ID
                    reports_tab_id = "__tab_TabContainerReportViewer_TabPanelReporting"
                    self.driver.execute_script(f"document.getElementById('{reports_tab_id}').click();")
                    time.sleep(2)
                    
                    # Then, click the Filing Inquiries tab
                    filing_tab_id = "__tab_TabContainerReportViewer_TabPanelReporting_TabContainerReports_TabPanelFilingInquiries"
                    self.driver.execute_script(f"document.getElementById('{filing_tab_id}').click();")
                    time.sleep(2)
                    
                    print("Used JavaScript to click tabs")
                except Exception as js_error:
                    print(f"JavaScript approach failed: {str(js_error)}")
                    
                    # Take screenshots for debugging
                    tab_debug_screenshot = os.path.join(self.download_dir, "tab_error.png")
                    self.driver.save_screenshot(tab_debug_screenshot)
                    print(f"Saved error state screenshot to {tab_debug_screenshot}")
                    
                    # Dump page source for debugging
                    with open(os.path.join(self.download_dir, "page_source.html"), "w", encoding="utf-8") as f:
                        f.write(self.driver.page_source)
                    
                    # Print available tabs
                    print("Available tabs on page:")
                    tabs = self.driver.find_elements(By.TAG_NAME, "a")
                    for tab in tabs:
                        if tab.get_attribute('id') and 'tab' in tab.get_attribute('id'):
                            print(f"Tab ID: {tab.get_attribute('id')}, Text: {tab.text}")
                    
                    raise Exception("Unable to navigate to Filing Inquiries tab")
            
            # Take a screenshot after tab navigation for debugging
            tab_screenshot = os.path.join(self.download_dir, "after_tab_click.png")
            self.driver.save_screenshot(tab_screenshot)
            print(f"Saved post-tab navigation screenshot to {tab_screenshot}")
            
            # Give the tab time to fully load
            time.sleep(5)
            
            # Step 4: Set up query parameters
            print("Setting up query parameters...")
            
            # Set Report Type to "Transactions" - THIS WILL TRIGGER A POSTBACK
            try:
                # Wait for the Report Type dropdown to be active
                report_type_dropdown_id = "TabContainerReportViewer_TabPanelReporting_TabContainerReports_TabPanelFilingInquiries_ddlReportType"
                report_type_dropdown = self.wait.until(EC.presence_of_element_located(
                    (By.ID, report_type_dropdown_id)))
                
                # Check if the dropdown has options
                select = Select(report_type_dropdown)
                if len(select.options) == 0 or "Transactions" not in [o.text for o in select.options]:
                    # Wait a bit and try again
                    time.sleep(3)
                    select = Select(report_type_dropdown)
                
                # Log available options
                print("Report Type options:")
                for option in select.options:
                    print(f" - {option.text}")
                
                # Select Transactions
                select.select_by_visible_text("Transactions")
                print("Set Report Type to Transactions")
                
                # Wait for the postback to complete
                time.sleep(5)  # First, simple wait
                try:
                    self.wait_for_postback()
                except:
                    pass  # Ignore errors in postback detection
                
                # Take a screenshot after selecting Report Type
                self.driver.save_screenshot(os.path.join(self.download_dir, "after_report_type.png"))
                
            except Exception as e:
                print(f"Error setting Report Type: {str(e)}")
                # Take a screenshot for debugging
                self.driver.save_screenshot(os.path.join(self.download_dir, "report_type_error.png"))
                
                # Try to find all select elements and see if any have Transactions
                print("Looking for Report Type dropdown...")
                select_elements = self.driver.find_elements(By.TAG_NAME, "select")
                for select_elem in select_elements:
                    try:
                        select_id = select_elem.get_attribute('id')
                        if select_id:
                            print(f"Found select element: {select_id}")
                            options = [option.text for option in Select(select_elem).options]
                            print(f"  Options: {options}")
                            
                            if 'Transactions' in options:
                                Select(select_elem).select_by_visible_text("Transactions")
                                print(f"Selected Transactions in {select_id}")
                                time.sleep(5)  # Wait for potential postback
                                break
                    except:
                        continue
                
                # Take a screenshot after trying to set Report Type
                self.driver.save_screenshot(os.path.join(self.download_dir, "after_report_type_recovery.png"))
            
            # Set By to "BA and HUB" (this dropdown appears after selecting Transactions)
            try:
                # Wait for the By dropdown to appear and be active (it's loaded dynamically)
                print("Waiting for 'By' dropdown to appear...")
                time.sleep(3)  # Wait for potential UI updates
                
                by_dropdown_id = "TabContainerReportViewer_TabPanelReporting_TabContainerReports_TabPanelFilingInquiries_ddlBy"
                by_dropdown = self.wait_for_element_to_be_active((By.ID, by_dropdown_id), timeout=10)
                
                if not by_dropdown:
                    # Try to find it with a more general approach
                    print("Trying to find 'By' dropdown via XPath...")
                    by_dropdowns = self.driver.find_elements(By.XPATH, "//select[contains(@id, 'ddlBy')]")
                    
                    if by_dropdowns:
                        by_dropdown = by_dropdowns[0]
                        print(f"Found By dropdown: {by_dropdown.get_attribute('id')}")
                    else:
                        # Try to find any dropdown that came after the Report Type
                        select_elements = self.driver.find_elements(By.TAG_NAME, "select")
                        for select_elem in select_elements:
                            select_id = select_elem.get_attribute('id')
                            if select_id and select_id != report_type_dropdown_id:
                                print(f"Found potential By dropdown: {select_id}")
                                by_dropdown = select_elem
                                break
                
                if by_dropdown:
                    # Get the options in the dropdown
                    select = Select(by_dropdown)
                    print("By dropdown options:")
                    for option in select.options:
                        print(f" - {option.text}")
                    
                    # Select BA and HUB
                    if "BA and HUB" in [o.text for o in select.options]:
                        select.select_by_visible_text("BA and HUB")
                        print("Set By to BA and HUB")
                        
                        # Wait for the postback to complete
                        time.sleep(5)  # Simple wait first
                        try:
                            self.wait_for_postback()
                        except:
                            pass  # Ignore errors in postback detection
                    else:
                        print("'BA and HUB' option not found in By dropdown")
                        # Select the first option as a fallback
                        if len(select.options) > 0:
                            select.select_by_index(1)  # Skip the first option if it's blank
                            print(f"Selected {select.options[1].text} from By dropdown")
                            time.sleep(5)  # Wait for potential postback
                else:
                    print("Could not find By dropdown")
                
                # Take a screenshot after selecting By
                self.driver.save_screenshot(os.path.join(self.download_dir, "after_by_selection.png"))
                
            except Exception as e:
                print(f"Error setting By dropdown: {str(e)}")
                self.driver.save_screenshot(os.path.join(self.download_dir, "by_dropdown_error.png"))
                
                # Print all select elements for debugging
                print("All select elements on page:")
                select_elements = self.driver.find_elements(By.TAG_NAME, "select")
                for select_elem in select_elements:
                    try:
                        select_id = select_elem.get_attribute('id')
                        print(f"Select ID: {select_id}")
                        options = [option.text for option in Select(select_elem).options]
                        print(f"  Options: {options}")
                    except:
                        print(f"  Could not get options for {select_id}")
            
            # Set Report Period based on date logic
            try:
                # Allow some time for the Report Period dropdown to become active
                time.sleep(2)
                report_period = self.determine_report_period(start_date, end_date)
                print(f"Using report period: {report_period}")
                
                period_dropdown_id = "TabContainerReportViewer_TabPanelReporting_TabContainerReports_TabPanelFilingInquiries_ddlReportPeriod"
                period_dropdown = self.wait_for_element_to_be_active((By.ID, period_dropdown_id), timeout=10)
                
                if period_dropdown:
                    # Get all available options and print them for debugging
                    period_select = Select(period_dropdown)
                    print("Available report periods:")
                    for option in period_select.options:
                        print(f" - {option.text}")
                    
                    # Select the appropriate period
                    if report_period in [o.text for o in period_select.options]:
                        period_select.select_by_visible_text(report_period)
                        print(f"Set Report Period to {report_period}")
                    else:
                        print(f"Report period {report_period} not found. Selecting first available period.")
                        # Find the first real option (skip any "Select One" type options)
                        for option in period_select.options:
                            if option.text and not option.text.startswith('-'):
                                period_select.select_by_visible_text(option.text)
                                print(f"Selected {option.text} as report period")
                                break
                    
                    # Wait for the postback to complete
                    time.sleep(3)
                    try:
                        self.wait_for_postback()
                    except:
                        pass  # Ignore errors in postback detection
                else:
                    print("Report Period dropdown not active or not found")
                
                # Take a screenshot after setting Report Period
                self.driver.save_screenshot(os.path.join(self.download_dir, "after_report_period.png"))
                
            except Exception as e:
                print(f"Error setting Report Period: {str(e)}")
                self.driver.save_screenshot(os.path.join(self.download_dir, "report_period_error.png"))
            
            # Set Balancing Authority
            try:
                # Allow some time for the Balancing Authority dropdown to become active
                time.sleep(2)
                authority_dropdown_id = "TabContainerReportViewer_TabPanelReporting_TabContainerReports_TabPanelFilingInquiries_ddlBalancingAuthority"
                authority_dropdown = self.wait_for_element_to_be_active((By.ID, authority_dropdown_id), timeout=10)
                
                if authority_dropdown:
                    authority_select = Select(authority_dropdown)
                    
                    # Get all available options and print them for debugging
                    print("Available authorities:")
                    available_authorities = [option.text for option in authority_select.options if option.text.strip()]
                    for auth in available_authorities:
                        print(f" - {auth}")
                    
                    # Check if requested authority is available
                    if authority in available_authorities:
                        authority_select.select_by_visible_text(authority)
                        print(f"Set Balancing Authority to {authority}")
                    else:
                        print(f"Warning: Requested authority '{authority}' not found. Using first available option.")
                        if available_authorities:
                            authority_select.select_by_visible_text(available_authorities[0])
                    
                    # Wait for the postback to complete
                    time.sleep(3)
                    try:
                        self.wait_for_postback()
                    except:
                        pass  # Ignore errors in postback detection
                else:
                    print("Balancing Authority dropdown not active or not found")
                
                # Take a screenshot after setting Balancing Authority
                self.driver.save_screenshot(os.path.join(self.download_dir, "after_authority.png"))
                
            except Exception as e:
                print(f"Error setting Balancing Authority: {str(e)}")
                self.driver.save_screenshot(os.path.join(self.download_dir, "authority_error.png"))
            
            # Set Start Date
            try:
                start_date_field_id = "TabContainerReportViewer_TabPanelReporting_TabContainerReports_TabPanelFilingInquiries_txtStartDate"
                start_date_field = self.wait.until(EC.presence_of_element_located(
                    (By.ID, start_date_field_id)))
                start_date_field.clear()
                start_date_field.send_keys(start_date)
                print(f"Set Start Date to {start_date}")
            except Exception as e:
                print(f"Error setting Start Date: {str(e)}")
                self.driver.save_screenshot(os.path.join(self.download_dir, "start_date_error.png"))
                
                # Try to find any input fields that might be for start date
                print("Looking for Start Date field...")
                input_elements = self.driver.find_elements(By.TAG_NAME, "input")
                for input_elem in input_elements:
                    input_id = input_elem.get_attribute('id')
                    if input_id and ('StartDate' in input_id or 'start' in input_id.lower()):
                        print(f"Found potential Start Date field: {input_id}")
                        try:
                            input_elem.clear()
                            input_elem.send_keys(start_date)
                            print(f"Set {start_date} in {input_id}")
                            break
                        except:
                            print(f"  Could not set date in {input_id}")
            
            # Set End Date
            try:
                end_date_field_id = "TabContainerReportViewer_TabPanelReporting_TabContainerReports_TabPanelFilingInquiries_txtEndDate"
                end_date_field = self.wait.until(EC.presence_of_element_located(
                    (By.ID, end_date_field_id)))
                end_date_field.clear()
                end_date_field.send_keys(end_date)
                print(f"Set End Date to {end_date}")
            except Exception as e:
                print(f"Error setting End Date: {str(e)}")
                self.driver.save_screenshot(os.path.join(self.download_dir, "end_date_error.png"))
                
                # Try to find any input fields that might be for end date
                print("Looking for End Date field...")
                input_elements = self.driver.find_elements(By.TAG_NAME, "input")
                for input_elem in input_elements:
                    input_id = input_elem.get_attribute('id')
                    if input_id and ('EndDate' in input_id or 'end' in input_id.lower()):
                        print(f"Found potential End Date field: {input_id}")
                        try:
                            input_elem.clear()
                            input_elem.send_keys(end_date)
                            print(f"Set {end_date} in {input_id}")
                            break
                        except:
                            print(f"  Could not set date in {input_id}")
            
            # Get seller dropdown
            try:
                # Give some time for seller dropdown to load/become active
                time.sleep(2)
                seller_dropdown_id = "TabContainerReportViewer_TabPanelReporting_TabContainerReports_TabPanelFilingInquiries_ddlSeller"
                seller_dropdown = self.wait_for_element_to_be_active((By.ID, seller_dropdown_id), timeout=10)
                
                if seller_dropdown:
                    seller_select = Select(seller_dropdown)
                    
                    # Determine sellers to process
                    if seller == "all":
                        sellers = [option.text for option in seller_select.options if option.text.strip()]
                        print(f"Found {len(sellers)} sellers to process")
                    else:
                        sellers = [seller]
                        print(f"Processing single seller: {seller}")
                else:
                    print("Seller dropdown not active or not found")
                    sellers = [seller] if seller != "all" else ["Default Seller"]
                
            except Exception as e:
                print(f"Error getting seller dropdown: {str(e)}")
                self.driver.save_screenshot(os.path.join(self.download_dir, "seller_dropdown_error.png"))
                
                # Try to find all select elements with 'Seller' in their ID
                print("Looking for Seller dropdown...")
                sellers = []
                select_elements = self.driver.find_elements(By.TAG_NAME, "select")
                for select_elem in select_elements:
                    select_id = select_elem.get_attribute('id')
                    if select_id and 'Seller' in select_id:
                        print(f"Found potential Seller dropdown: {select_id}")
                        try:
                            options = [option.text for option in Select(select_elem).options if option.text.strip()]
                            print(f"  Found {len(options)} seller options")
                            
                            if seller == "all":
                                sellers = options
                            else:
                                if seller in options:
                                    sellers = [seller]
                                else:
                                    print(f"Warning: Seller '{seller}' not found in dropdown")
                                    sellers = [options[0]]
                            
                            # We found a dropdown, but don't select anything yet - we'll do that in the loop
                            seller_dropdown = select_elem
                            seller_select = Select(seller_dropdown)
                            break
                        except:
                            print(f"  Could not get options from {select_id}")
                
                if not sellers:
                    print("Could not find seller dropdown, using default value")
                    sellers = [seller] if seller != "all" else ["Default Seller"]
            
            # Set CSV export format
            try:
                export_dropdown_id = "TabContainerReportViewer_TabPanelReporting_TabContainerReports_TabPanelFilingInquiries_ddlExport"
                export_dropdown = self.wait.until(EC.presence_of_element_located(
                    (By.ID, export_dropdown_id)))
                Select(export_dropdown).select_by_visible_text("CSV")
                print("Set Export format to CSV")
            except Exception as e:
                print(f"Error setting Export format: {str(e)}")
                self.driver.save_screenshot(os.path.join(self.download_dir, "export_format_error.png"))
                
                # Try to find all select elements with 'Export' in their ID
                print("Looking for Export format dropdown...")
                select_elements = self.driver.find_elements(By.TAG_NAME, "select")
                for select_elem in select_elements:
                    select_id = select_elem.get_attribute('id')
                    if select_id and 'Export' in select_id:
                        print(f"Found potential Export dropdown: {select_id}")
                        try:
                            options = [option.text for option in Select(select_elem).options]
                            print(f"  Options: {options}")
                            
                            # Try selecting CSV if available
                            if 'CSV' in options:
                                Select(select_elem).select_by_visible_text("CSV")
                                print(f"Selected CSV in {select_id}")
                                time.sleep(2)
                                break
                        except:
                            print(f"  Could not select from {select_id}")
            
            # Take a screenshot after setting all parameters
            self.driver.save_screenshot(os.path.join(self.download_dir, "before_submit.png"))
            
            # Process each seller
            success_count = 0
            for idx, current_seller in enumerate(sellers):
                try:
                    print(f"Processing seller {idx+1}/{len(sellers)}: {current_seller}")
                    
                    # Select the current seller
                    try:
                        if hasattr(self, 'seller_select'):
                            self.seller_select.select_by_visible_text(current_seller)
                        else:
                            Select(seller_dropdown).select_by_visible_text(current_seller)
                        print(f"Selected seller: {current_seller}")
                        time.sleep(2)  # Give the UI time to update
                    except Exception as seller_error:
                        print(f"Error selecting seller {current_seller}: {str(seller_error)}")
                        continue
                    
                    # Click Submit button - first try btnSubmitOptional based on HTML
                    submit_button = None
                    try:
                        # First try the exact button ID from the HTML
                        submit_button_id = "TabContainerReportViewer_TabPanelReporting_TabContainerReports_TabPanelFilingInquiries_btnSubmitOptional"
                        submit_button = self.driver.find_element(By.ID, submit_button_id)
                        # Check if the button is enabled
                        if not submit_button.is_enabled() or "aspNetDisabled" in submit_button.get_attribute("class"):
                            print("Submit button is disabled, looking for other buttons...")
                            submit_button = None
                        else:
                            submit_button.click()
                            print("Clicked Submit button (btnSubmitOptional)")
                    except Exception as btn_error:
                        print(f"Error finding btnSubmitOptional: {str(btn_error)}")
                        submit_button = None
                    
                    # If the primary submit button didn't work, try alternatives
                    if not submit_button:
                        try:
                            # Try different possible button selectors
                            possible_selectors = [
                                "//input[contains(@id, 'btnSubmit')]",
                                "//button[contains(@id, 'btnSubmit')]",
                                "//input[@value='Submit']",
                                "//button[text()='Submit']",
                                "//input[@type='submit']",
                                "//input[contains(@id, 'btnRun')]",
                                "//input[contains(@id, 'btnExecute')]"
                            ]
                            
                            for selector in possible_selectors:
                                try:
                                    submit_buttons = self.driver.find_elements(By.XPATH, selector)
                                    if submit_buttons:
                                        submit_button = submit_buttons[0]
                                        for btn in submit_buttons:
                                            # Skip disabled buttons
                                            if not btn.is_enabled() or "aspNetDisabled" in btn.get_attribute("class"):
                                                continue
                                                
                                            btn_value = btn.get_attribute('value') or btn.text
                                            print(f"Found button: {btn.get_attribute('id')} - {btn_value}")
                                            if 'Submit' in btn_value or 'Run' in btn_value or 'Execute' in btn_value:
                                                submit_button = btn
                                                break
                                        
                                        # Click the best matching button
                                        if submit_button and submit_button.is_enabled():
                                            submit_button.click()
                                            print(f"Clicked button: {submit_button.get_attribute('id')}")
                                            break
                                except NoSuchElementException:
                                    continue
                            
                            if not submit_button:
                                # Try JavaScript click on any submit button
                                print("Trying JavaScript click on Submit button...")
                                self.driver.execute_script("""
                                    var buttons = document.querySelectorAll('input[type=\"submit\"]:not([disabled]), button[type=\"submit\"]:not([disabled])');
                                    for (var i = 0; i < buttons.length; i++) {
                                        if (buttons[i].value.includes('Submit') || buttons[i].innerText.includes('Submit')) {
                                            buttons[i].click();
                                            return true;
                                        }
                                    }
                                    // If no Submit button, click any enabled submit button
                                    if (buttons.length > 0) {
                                        buttons[0].click();
                                        return true;
                                    }
                                    return false;
                                """)
                                print("Used JavaScript to click Submit button")
                        except Exception as alt_btn_error:
                            print(f"Error with alternative button approach: {str(alt_btn_error)}")
                            
                            # List all buttons on the page
                            print("All buttons on the page:")
                            all_buttons = self.driver.find_elements(By.XPATH, "//input[@type='submit'] | //button | //input[@type='button']")
                            for btn in all_buttons:
                                btn_id = btn.get_attribute('id')
                                btn_value = btn.get_attribute('value') or btn.text
                                btn_type = btn.get_attribute('type')
                                btn_enabled = btn.is_enabled() and "aspNetDisabled" not in btn.get_attribute("class")
                                print(f"Button: ID={btn_id}, Value={btn_value}, Type={btn_type}, Enabled={btn_enabled}")
                                
                                # Try clicking each button that seems like a submit button and is enabled
                                if btn_enabled and ('Submit' in btn_value or 'Run' in btn_value or 'Execute' in btn_value or 
                                    'submit' in btn_type or ('Submit' in btn_id) or ('Run' in btn_id)):
                                    try:
                                        btn.click()
                                        print(f"Clicked button: {btn_id}")
                                        break
                                    except:
                                        print(f"Could not click button: {btn_id}")
                            
                            # If we still couldn't find a button, try one last JavaScript approach
                            try:
                                print("Trying form submission via JavaScript...")
                                self.driver.execute_script("""
                                    var forms = document.forms;
                                    if (forms.length > 0) forms[0].submit();
                                """)
                                print("Submitted form via JavaScript")
                            except Exception as js_error:
                                print(f"JavaScript form submission failed: {str(js_error)}")
                    
                    # Wait for download to complete (monitoring download directory)
                    print("Waiting for download to complete...")
                    download_successful = self._wait_for_download(45)  # 45 second timeout
                    
                    if download_successful:
                        print(f"Successfully downloaded data for seller: {current_seller}")
                        success_count += 1
                    else:
                        print(f"Download may have failed for seller: {current_seller}")
                        self.driver.save_screenshot(os.path.join(self.download_dir, f"after_submit_{idx}.png"))
                    
                    # Brief pause between sellers
                    time.sleep(3)
                    
                except Exception as e:
                    print(f"Error processing seller {current_seller}: {str(e)}")
                    self.driver.save_screenshot(os.path.join(self.download_dir, f"seller_error_{idx}.png"))
            
            print(f"Download process completed. Successfully downloaded data for {success_count}/{len(sellers)} sellers.")
            return success_count > 0
            
        except Exception as e:
            print(f"Error in download process: {str(e)}")
            self.driver.save_screenshot(os.path.join(self.download_dir, "final_error.png"))
            return False
    
    def _wait_for_download(self, timeout=45):
        """
        Wait for a download to complete by monitoring the download directory.
        
        Args:
            timeout (int): Maximum time to wait in seconds
            
        Returns:
            bool: True if a new file was detected, False otherwise
        """
        # Get list of files before download
        before_download = set(os.listdir(self.download_dir))
        
        # Wait for new file to appear
        start_time = time.time()
        while time.time() - start_time < timeout:
            current_files = set(os.listdir(self.download_dir))
            new_files = current_files - before_download
            
            # Check for temporary download files (Chrome adds .crdownload extension)
            temp_downloads = [f for f in new_files if f.endswith('.crdownload') or f.endswith('.tmp')]
            
            if new_files and not temp_downloads:
                print(f"Detected new file(s): {new_files}")
                return True
                
            time.sleep(0.5)
        
        print(f"No new files detected after {timeout} seconds")
        return False
    
    def close(self):
        """Close the browser"""
        if hasattr(self, 'driver'):
            self.driver.quit()
    
    def __del__(self):
        """Ensure browser is closed when object is deleted"""
        try:
            self.close()
        except:
            pass


def main():
    """Main function to handle user input and run the downloader"""
    print("FERC EQR Report Viewer Downloader")
    print("=================================")
    
    # Check if --default flag is provided
    if "--default" in sys.argv:
        print("Using default values for all parameters")
        # Default to current quarter
        today = datetime.now()
        first_day = datetime(today.year, ((today.month-1)//3)*3+1, 1)
        start_date = first_day.strftime("%m/%d/%Y")
        end_date = today.strftime("%m/%d/%Y")
        seller = "all"
        authority = "CISO"
        download_dir = os.getcwd()
        headless = False
    else:
        # Interactive mode
        while True:
            start_date = input("Enter start date (MM/DD/YYYY): ")
            try:
                datetime.strptime(start_date, "%m/%d/%Y")
                break
            except ValueError:
                print("Invalid date format. Please use MM/DD/YYYY.")
        
        while True:
            end_date = input("Enter end date (MM/DD/YYYY): ")
            try:
                datetime.strptime(end_date, "%m/%d/%Y")
                break
            except ValueError:
                print("Invalid date format. Please use MM/DD/YYYY.")

        seller = input("Enter seller name (or just hit enter for all sellers): ")
        if not seller:
            seller = "all"
        
        authority = input("Enter authority (default is CISO): ")
        if not authority:
            authority = "CISO"
        
        download_dir = input("Enter download directory (or press Enter for current directory): ")
        if not download_dir:
            download_dir = os.getcwd()
        elif not os.path.exists(download_dir):
            print(f"Directory {download_dir} does not exist. Creating it...")
            os.makedirs(download_dir)
        
        headless = input("Run in headless mode? (y/n, default: n): ").lower() == 'y'
    
    print("\nStarting downloader with the following parameters:")
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Seller: {seller}")
    print(f"Authority: {authority}")
    print(f"Download Directory: {download_dir}")
    print(f"Headless Mode: {headless}")
    print("\n")
    
    # Create downloader and run
    try:
        downloader = FercDownloader(download_dir, headless)
        try:
            result = downloader.download_transaction_data(start_date, end_date, seller, authority)
            
            if result:
                print("Data download completed successfully!")
            else:
                print("Data download failed. Please check the logs for details.")
        finally:
            # Ask user if they want to close the browser
            if input("\nClose browser? (y/n, default: y): ").lower() != 'n':
                downloader.close()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        print("Please ensure Chrome is installed and up to date.")


if __name__ == "__main__":
    main()