"""Factory returning proper parser"""

from .bayer import ParserBayerRG
from ..image.color_format import PixelPlane, PixelFormat
from .rgb import ParserARGB, ParserRGBA
from .yuv import ParserYUV420, ParserYUV420Planar, ParserYUV422, ParserYUV422Planar
from .grayscale import ParserGrayscale


class ParserFactory:
    """Parser factory"""

    @staticmethod
    def create_object(color_format):
        """Get parser for provided color format.

        Keyword arguments:
            color_format: instance of ColorFormat

        Returns: instance of parser
        """
        mapping = {}
        if color_format.pixel_plane == PixelPlane.PACKED:
            mapping = {
                PixelFormat.BAYER_RG: ParserBayerRG,
                PixelFormat.MONO: ParserGrayscale,
                PixelFormat.RGBA: ParserRGBA,
                PixelFormat.BGRA: ParserRGBA,
                PixelFormat.ARGB: ParserARGB,
                PixelFormat.ABGR: ParserARGB,
                PixelFormat.YUYV: ParserYUV422,
                PixelFormat.UYVY: ParserYUV422,
                PixelFormat.VYUY: ParserYUV422,
                PixelFormat.YVYU: ParserYUV422,
            }
        elif color_format.pixel_plane == PixelPlane.SEMIPLANAR:
            mapping = {
                PixelFormat.YUV: ParserYUV420,
                PixelFormat.YVU: ParserYUV420,
            }
        elif color_format.pixel_plane == PixelPlane.PLANAR:
            if color_format.subsampling_vertical == 1:
                mapping = {PixelFormat.YUV: ParserYUV422Planar}
            else:
                mapping = {
                    PixelFormat.YUV: ParserYUV420Planar,
                    PixelFormat.YVU: ParserYUV420Planar,
                }

        proper_class = mapping.get(color_format.pixel_format)
        if proper_class is None:
            raise NotImplementedError(
                f"No parser found for {color_format.name} color format"
            )
        return proper_class()
