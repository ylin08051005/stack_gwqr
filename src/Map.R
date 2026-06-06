library(dplyr)
library(readr)
library(sf)
library(tmap)
library(showtext)

font_add("PingFang", "/System/Library/Fonts/PingFang.ttc")
showtext_auto()
my_font <- "PingFang"

vgdf <- st_read("/Users/ylin/Documents/stack_gwqr/data/TWD97/VILLAGE_NLSC_1140825.shp")
taipei <- vgdf %>% 
  filter(COUNTYNAME %in% c("臺北市", "台北市")) %>%
  st_make_valid()

# 針對行政區標籤進行微調位移
town_boundary <- taipei %>%
  group_by(TOWNNAME) %>% 
  summarise(.groups = "drop") %>%
  mutate(
    shift_x = case_when(
      TOWNNAME == "中山區" ~ -0.70,
      TOWNNAME == "中正區" ~ -0.50, 
      TRUE ~ 0
    ),
    shift_y = case_when(
      TOWNNAME == "中山區" ~ -0.800,
      TOWNNAME == "中正區" ~ -0.100, 
      TRUE ~ 0
    )
  )

file_paths <- c(
  xgboost = "/Users/ylin/Documents/stack_gwqr/results/predictions/113_xgboost_L3_test_predictions.csv",
  rf      = "/Users/ylin/Documents/stack_gwqr/results/predictions/113_rf_L3_test_predictions.csv",
  lgb     = "/Users/ylin/Documents/stack_gwqr/results/predictions/113_lgb_L3_test_predictions.csv",
  nn      = "/Users/ylin/Documents/stack_gwqr/results/predictions/113_nn_L3_test_predictions.csv"
)


#file_paths <- c(
 #xgboost = "/Users/ylin/Documents/stack_gwqr/results/112_pred/112_xgboost_L3_test_predictions.csv",
#  rf      = "/Users/ylin/Documents/stack_gwqr/results/112_pred/112_rf_L1_test_predictions.csv",
# lgb     = "/Users/ylin/Documents/stack_gwqr/results/112_pred/112_lgb_L1_test_predictions.csv",
# nn      = "/Users/ylin/Documents/stack_gwqr/results/112_pred/112_nn_L1_test_predictions.csv"
#)

all_predictions <- lapply(file_paths, read_csv, show_col_types = FALSE) %>% bind_rows()

all_points_sf <- st_as_sf(all_predictions, coords = c("經度", "緯度"), crs = 4326) %>%
  st_transform(3826)

tmap_mode("plot")

tm_shape(taipei) +
  tm_polygons(
    fill = "TOWNNAME", 
    fill.scale = tm_scale_categorical(values = "Pastel2"),
    fill.legend = tm_legend(title = "行政區域"),
    fill_alpha = 0.5,
    border.col = "white"
  ) +
  tm_shape(all_points_sf) +
  tm_dots(
    fill = "#984EA3",         
    size = 0.15,
    fill_alpha = 0.7,
    fill.legend = tm_legend(title = "住宅交易點位 (L3 測試集)")
  ) +
  tm_shape(town_boundary) +
  tm_borders(col = "#4A4A4A", lwd = 1.2) +
  tm_text(
    "TOWNNAME", 
    size = 0.7, 
    fontface = "bold", 
    fontfamily = my_font,
    xmod = "shift_x",            
    bg.color = "white",         
    bg.alpha = 0.5
  ) +
  tm_title("113年台北市住宅交易分布圖 (L3層級測試集)", fontfamily = my_font, size = 1.2) +
  tm_layout(
    legend.fontfamily = my_font,
    legend.outside = TRUE,
    legend.outside.position = "right",
    frame = FALSE,
    inner.margins = c(0, 0, 0, 0), 
    outer.margins = c(0.02, 0.02, 0.02, 0.02)
  ) +
  tm_scalebar(position = c("left", "bottom")) + 
  tm_compass(type = "8star", position = c("right", "top"), size = 1.5)

