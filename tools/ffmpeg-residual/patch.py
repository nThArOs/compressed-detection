"""Make the FFmpeg H.264 decoder dump the luma residual of every inter macroblock.

Per decoded frame, written to $H264_RESIDUAL_OUT right after the macroblocks are reconstructed (before the
deblocking filter): an int16 plane of (reconstruction - motion-compensated prediction), then a uint8 plane with
one flag per macroblock, 1 for intra macroblocks (no prediction to subtract, residual left at 0).
Planes are padded to whole macroblocks. Frames come in decoding order: decode with -threads 1.
"""
import sys
from pathlib import Path

src = Path(sys.argv[1]) / "libavcodec"


def edit(name, old, new):
    p = src / name
    text = p.read_text()
    assert text.count(old) == 1, (name, old)
    p.write_text(text.replace(old, new))


edit("h264_mb.c", '#include "threadframe.h"\n', '''#include "threadframe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int16_t *g_res;
static uint8_t *g_intra;
static int g_w, g_h;

static void res_save_pred(uint8_t pred[16][16], const uint8_t *dest, ptrdiff_t ls)
{
    for (int y = 0; y < 16; y++)
        memcpy(pred[y], dest + y * ls, 16);
}

static void res_store(const H264Context *h, int mb_x, int mb_y, uint8_t pred[16][16], const uint8_t *dest, ptrdiff_t ls)
{
    int w = h->mb_width * 16, hh = h->mb_height * 16;
    if (w != g_w || hh != g_h) {
        free(g_res);
        free(g_intra);
        g_res = calloc((size_t)w * hh, sizeof(*g_res));
        g_intra = calloc((size_t)h->mb_width * h->mb_height, 1);
        g_w = w;
        g_h = hh;
    }
    if (!g_res || mb_x >= h->mb_width || mb_y >= h->mb_height)
        return;
    if (!pred) {
        g_intra[mb_y * h->mb_width + mb_x] = 1;
        return;
    }
    for (int y = 0; y < 16; y++)
        for (int x = 0; x < 16; x++)
            g_res[(mb_y * 16 + y) * w + mb_x * 16 + x] = dest[y * ls + x] - pred[y][x];
}

void ff_h264_residual_flush(const H264Context *h)
{
    static FILE *out;
    const char *path = getenv("H264_RESIDUAL_OUT");
    if (!path || !g_res)
        return;
    if (!out)
        out = fopen(path, "wb");
    if (!out)
        return;
    fwrite(g_res, sizeof(*g_res), (size_t)g_w * g_h, out);
    fwrite(g_intra, 1, (size_t)(g_w / 16) * (g_h / 16), out);
    fflush(out);
    memset(g_res, 0, (size_t)g_w * g_h * sizeof(*g_res));
    memset(g_intra, 0, (size_t)(g_w / 16) * (g_h / 16));
}
''')

edit("h264_mb_template.c", '''        hl_decode_mb_idct_luma(h, sl, mb_type, SIMPLE, transform_bypass,
                               PIXEL_SHIFT, block_offset, linesize, dest_y, 0);

        if ((SIMPLE || !CONFIG_GRAY || !(h->flags & AV_CODEC_FLAG_GRAY)) &&
            (sl->cbp & 0x30)) {
            uint8_t *dest[2] = { dest_cb, dest_cr };
            if (transform_bypass) {''', '''#if BITS == 8
        uint8_t res_pred[16][16];
        if (!IS_INTRA(mb_type))
            res_save_pred(res_pred, dest_y, linesize);
#endif
        hl_decode_mb_idct_luma(h, sl, mb_type, SIMPLE, transform_bypass,
                               PIXEL_SHIFT, block_offset, linesize, dest_y, 0);
#if BITS == 8
        res_store(h, sl->mb_x, sl->mb_y, IS_INTRA(mb_type) ? NULL : res_pred, dest_y, linesize);
#endif

        if ((SIMPLE || !CONFIG_GRAY || !(h->flags & AV_CODEC_FLAG_GRAY)) &&
            (sl->cbp & 0x30)) {
            uint8_t *dest[2] = { dest_cb, dest_cr };
            if (transform_bypass) {''')

edit("h264dec.h", "int ff_h264_field_end(H264Context *h, H264SliceContext *sl, int in_setup);",
     "int ff_h264_field_end(H264Context *h, H264SliceContext *sl, int in_setup);\nvoid ff_h264_residual_flush(const H264Context *h);")
edit("h264_picture.c", "    if (!in_setup && !h->droppable)\n        ff_thread_report_progress(&cur->tf, INT_MAX,",
     "    if (!in_setup)\n        ff_h264_residual_flush(h);\n\n    if (!in_setup && !h->droppable)\n        ff_thread_report_progress(&cur->tf, INT_MAX,")
