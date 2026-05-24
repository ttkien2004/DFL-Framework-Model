from config import Config
import math

class LDP:
    @staticmethod
    def get_ldp_sigma():
        """
        Tính độ lệch chuẩn của nhiễu dựa trên công thức Gaussian Mechanism:
        sigma = (C * sqrt(2 * ln(1.25 / delta))) / epsilon
        """
        if not Config.ENABLE_LDP:
            return 0.0
            
        numerator = Config.LDP_CLIPPING_THRESHOLD * math.sqrt(2 * math.log(1.25 / Config.LDP_DELTA))
        sigma = numerator / Config.LDP_EPSILON
        
        return sigma