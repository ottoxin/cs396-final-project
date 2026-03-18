TEX_FILE := carm_proposal.tex
ifneq ($(origin TEX), default)
ifneq ($(origin TEX), undefined)
TEX_FILE := $(TEX)
endif
endif

BUILD := ./build_tex.sh

.PHONY: pdf clean distclean zotero-sync pdf-sync

pdf:
	$(BUILD) $(TEX_FILE)

zotero-sync:
	./sync_zotero_bib.sh

pdf-sync:
	ZOTERO_SYNC=1 $(BUILD) $(TEX_FILE)

clean:
	$(BUILD) --clean $(TEX_FILE)

distclean:
	$(BUILD) --distclean $(TEX_FILE)
