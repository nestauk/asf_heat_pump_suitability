"""
Create a sample of buildings containing flats for manual labelling to use in model training.
"""

# STEP-BY-STEP APPROACH
# 1. Label all UPRNs as flats or not.
# 2. Identify all buildings containing flats.
# 3. Label buildings with rural/urban indicator
# 4. Label buildings with IMD decile
# 5. Label buildings with country
# 7. Get count of flats per building
# 8. Label buildings with grouped-count
# 9. Sample based on rural/urban indicator; IMD decile; country; and grouped count
# 10. Enrich sample with additional data:
# - URL
# - Count of UPRNs per building
# 11. Convert to kml file
