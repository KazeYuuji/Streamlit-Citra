import numpy as np


def estimate_volume_from_mask(
    mask: np.ndarray,
    pixel_to_cm_ratio: float = 0.05,
    avg_thickness_cm: float = 0.3,
    method: str = "pixel_count",
) -> dict:
    """Estimate the volume of detected plastic waste from a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (255 = plastic, 0 = background).
    pixel_to_cm_ratio : float
        Conversion factor: length of one pixel side in centimetres.
        Default 0.05 cm/pixel (~0.5 mm per px for a typical drone/river image).
    avg_thickness_cm : float
        Assumed average thickness of floating plastic waste in cm.
        Default 0.3 cm (≈ 3 mm, typical for plastic bags/bottles).
    method : str
        - 'pixel_count' : area = pixel_count * px_ratio², volume = area * thickness
        - 'contour_area': use cv2 contour area (more accurate for irregular shapes)

    Returns
    -------
    dict with keys:
        area_cm2, area_m2, volume_cm3, volume_m3, volume_liters,
        estimated_mass_kg, pixel_to_cm_ratio, avg_thickness_cm
    """
    if method == "contour_area":
        import cv2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pixel_area = sum(cv2.contourArea(c) for c in contours)
    else:
        pixel_area = float(np.sum(mask == 255))

    px_area_cm2 = pixel_area * (pixel_to_cm_ratio ** 2)
    px_area_m2 = px_area_cm2 / 10000.0

    volume_cm3 = px_area_cm2 * avg_thickness_cm
    volume_m3 = volume_cm3 / 1_000_000.0
    volume_liters = volume_cm3 / 1000.0

    density_kg_per_m3 = 920.0
    estimated_mass_kg = volume_m3 * density_kg_per_m3

    return {
        "area_cm2": round(px_area_cm2, 3),
        "area_m2": round(px_area_m2, 6),
        "volume_cm3": round(volume_cm3, 3),
        "volume_m3": round(volume_m3, 9),
        "volume_liters": round(volume_liters, 3),
        "estimated_mass_kg": round(estimated_mass_kg, 6),
        "pixel_to_cm_ratio": pixel_to_cm_ratio,
        "avg_thickness_cm": avg_thickness_cm,
    }
