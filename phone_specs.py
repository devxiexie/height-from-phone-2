"""
values in millimetres
"""

PHONE_SPECS = {
    # --- Apple (newest first) ---
    "iPhone 17 Pro Max":        (163.4, 78.0),
    "iPhone 17 Pro":            (150.0, 71.9),
    "iPhone 17":                (149.6, 71.5),
    "iPhone 16 Pro Max":        (163.0, 77.6),
    "iPhone 16 Pro":            (149.6, 71.5),
    "iPhone 16 Plus":           (160.9, 77.8),
    "iPhone 16":                (147.6, 71.6),
    "iPhone 16e":               (146.7, 71.5),
    "iPhone 15 Pro Max":        (159.9, 76.7),
    "iPhone 15 Plus":           (160.9, 77.8),
    "iPhone 15 / 15 Pro":       (147.6, 71.6),
    "iPhone 14 / 13":           (146.7, 71.5),
    # --- Samsung (newest first) ---
    "Samsung Galaxy S26 Ultra": (163.6, 78.1),
    "Samsung Galaxy S25 Ultra": (162.8, 77.6),
    "Samsung Galaxy S25+":      (158.4, 75.8),
    "Samsung Galaxy S25":       (146.9, 70.5),
    "Samsung Galaxy S24 Ultra": (162.3, 79.0),
    "Samsung Galaxy S24":       (147.0, 70.6),
    "Samsung Galaxy S23":       (146.3, 70.9),
    "Samsung Galaxy A54":       (158.2, 76.7),
    # --- Google ---
    "Google Pixel 9 Pro XL":    (162.8, 76.6),
    "Google Pixel 9 Pro":       (152.8, 72.0),
    "Google Pixel 9":           (152.8, 72.0),
    "Google Pixel 8 Pro":       (162.6, 76.5),
    "Google Pixel 8":           (150.5, 70.8),
    # --- OnePlus ---
    "OnePlus 13":               (162.9, 76.5),
    "OnePlus 12":               (164.3, 75.8),
}

GENERIC_PHONE = (152.0, 72.0)

GENERIC_LABEL = "Generic phone (average ~152x72mm)"
CUSTOM_LABEL = "Custom (enter dimensions below)"


def dropdown_choices():
    return list(PHONE_SPECS.keys()) + [GENERIC_LABEL, CUSTOM_LABEL]


def resolve_phone_size(selection, custom_height_mm=None, custom_width_mm=None):
    if selection in PHONE_SPECS:
        h, w = PHONE_SPECS[selection]
        return h, w, selection
    if selection == CUSTOM_LABEL:
        has_w = custom_width_mm and custom_width_mm > 0
        has_h = custom_height_mm and custom_height_mm > 0
        if has_w or has_h:
            w = float(custom_width_mm) if has_w else GENERIC_PHONE[1]
            h = float(custom_height_mm) if has_h else GENERIC_PHONE[0]
            return h, w, f"custom ({w:.0f}mm wide)"
        return GENERIC_PHONE[0], GENERIC_PHONE[1], "generic (no custom value given)"
    return GENERIC_PHONE[0], GENERIC_PHONE[1], "generic"
