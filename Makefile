# =============================================================================
# cnn-ops-in-c — build all 109 standalone operator demos
# =============================================================================
# Usage:
#   make              — build every section into bin/
#   make test         — build + run all, report PASS/FAIL
#   make convolutions — build section 01 only
#   make activations  — build section 02 only
#   make pooling      — build section 03 only
#   make normalization— build section 04 only
#   make shape_ops    — build section 05 only
#   make linear       — build section 06 only
#   make recurrent    — build section 07 only
#   make losses       — build section 08 only
#   make postproc     — build section 09 only
#   make attention    — build section 10 only
#   make clean        — remove bin/
#   make run_01       — run all binaries in bin/01_convolutions/
#   make run_02 ... run_10  — same for each section

CC      := gcc
CFLAGS  := -O2 -Wall -Wextra -Wno-unused-result
LDFLAGS := -lm

SECTIONS := \
    01_convolutions \
    02_activations  \
    03_pooling      \
    04_normalization \
    05_shape_ops    \
    06_linear       \
    07_recurrent    \
    08_losses       \
    09_postprocessing \
    10_attention

# Collect all source files across all sections
ALL_SRCS := $(foreach s,$(SECTIONS),$(wildcard $(s)/*.c))

# Map each .c → bin/<section>/<stem>
bin_of = bin/$(dir $(1))$(basename $(notdir $(1)))
ALL_BINS := $(foreach src,$(ALL_SRCS),$(call bin_of,$(src)))

# -----------------------------------------------------------------------------
# Default target: build everything
# -----------------------------------------------------------------------------
.PHONY: all
all: $(ALL_BINS)

# Generic rule: compile any .c → bin/<section>/<stem>
bin/%: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Explicit static pattern so Make can match section/file.c → bin/section/file
$(foreach s,$(SECTIONS), \
  $(eval $(addprefix bin/$(s)/,$(basename $(notdir $(wildcard $(s)/*.c)))): bin/$(s)/%: $(s)/%.c ; \
    @mkdir -p bin/$(s) ; \
    $$(CC) $$(CFLAGS) $$< -o $$@ $$(LDFLAGS)))

# -----------------------------------------------------------------------------
# Section shortcuts
# -----------------------------------------------------------------------------
SECTION_BINS = $(foreach src,$(wildcard $(1)/*.c),$(call bin_of,$(src)))

.PHONY: convolutions
convolutions: $(call SECTION_BINS,01_convolutions)

.PHONY: activations
activations: $(call SECTION_BINS,02_activations)

.PHONY: pooling
pooling: $(call SECTION_BINS,03_pooling)

.PHONY: normalization
normalization: $(call SECTION_BINS,04_normalization)

.PHONY: shape_ops
shape_ops: $(call SECTION_BINS,05_shape_ops)

.PHONY: linear
linear: $(call SECTION_BINS,06_linear)

.PHONY: recurrent
recurrent: $(call SECTION_BINS,07_recurrent)

.PHONY: losses
losses: $(call SECTION_BINS,08_losses)

.PHONY: postproc
postproc: $(call SECTION_BINS,09_postprocessing)

.PHONY: attention
attention: $(call SECTION_BINS,10_attention)

# -----------------------------------------------------------------------------
# Run shortcuts — execute every binary in a section's bin dir
# -----------------------------------------------------------------------------
define run_section
.PHONY: run_$(1)
run_$(1):
	@echo "=== Running $(2) ==="
	@for f in bin/$(2)/*; do \
	    printf "  %-45s " "$$$$f"; \
	    if $$$$f > /dev/null 2>&1; then echo "OK"; else echo "FAIL"; fi; \
	done
endef

$(eval $(call run_section,01,01_convolutions))
$(eval $(call run_section,02,02_activations))
$(eval $(call run_section,03,03_pooling))
$(eval $(call run_section,04,04_normalization))
$(eval $(call run_section,05,05_shape_ops))
$(eval $(call run_section,06,06_linear))
$(eval $(call run_section,07,07_recurrent))
$(eval $(call run_section,08,08_losses))
$(eval $(call run_section,09,09_postprocessing))
$(eval $(call run_section,10,10_attention))

# -----------------------------------------------------------------------------
# test — build everything then run build_and_test.sh on every section
# -----------------------------------------------------------------------------
.PHONY: test
test:
	@echo ""
	@echo "============================================================"
	@echo "  cnn-ops-in-c  full test suite"
	@echo "============================================================"
	@TOTAL_PASS=0; TOTAL_FAIL=0; \
	for s in $(SECTIONS); do \
	    echo ""; \
	    echo "--- $$s ---"; \
	    result=$$(bash build_and_test.sh $$s 2>&1); \
	    echo "$$result" | grep -E "^(OK|FAIL)" | head -5; \
	    pass=$$(echo "$$result" | grep -c "^OK"); \
	    fail=$$(echo "$$result" | grep -c "^FAIL"); \
	    echo "  PASS=$$pass  FAIL=$$fail"; \
	    TOTAL_PASS=$$((TOTAL_PASS+pass)); \
	    TOTAL_FAIL=$$((TOTAL_FAIL+fail)); \
	done; \
	echo ""; \
	echo "============================================================"; \
	echo "  TOTAL  PASS=$$TOTAL_PASS  FAIL=$$TOTAL_FAIL"; \
	echo "============================================================"

# -----------------------------------------------------------------------------
# clean
# -----------------------------------------------------------------------------
.PHONY: clean
clean:
	rm -rf bin/
	@echo "Removed bin/"
