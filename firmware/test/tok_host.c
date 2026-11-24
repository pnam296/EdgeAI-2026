/* Tach tu bang code C cua firmware -> so voi Python (tools/export_intent.py goi).
 *   tok_host <seq_len> <file: moi dong 1 cau da normalize>  -> moi dong: token id cach nhau dau cach */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "logic.h"
#include "vocab.h"

int main(int argc, char **argv)
{
    if (argc < 3) return 2;
    int seq = atoi(argv[1]);
    FILE *f = fopen(argv[2], "r");
    if (!f) return 1;
    char line[1024];
    int32_t ids[MAX_SEQ];
    while (fgets(line, sizeof line, f)) {
        line[strcspn(line, "\r\n")] = 0;
        normalize_text(line);
        tokenize(line, vocab, vocab_len, ids, seq);
        for (int i = 0; i < seq; i++) printf(i ? " %d" : "%d", ids[i]);
        printf("\n");
    }
    fclose(f);
    return 0;
}
