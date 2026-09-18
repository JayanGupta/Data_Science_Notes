# Sprint 4 \u2014 DAX Measures Reference

Copy-paste these into Power BI: **Modeling tab \u2192 New Measure**, with `products_for_powerbi` loaded as a table (see README.md for the connection steps).

All measures below assume the table is named `products` in Power BI's data pane. If you named it something else on import, replace `products` throughout.

---

## The Brief (write this down before opening Power BI)

**Audience:** Category Manager (Electronics)

1. Which categories/brands are priced above the category average?
2. Where is customer satisfaction (rating) weakest relative to price?
3. How does the price/rating relationship shift when filtered by price range?

Every measure and visual below exists to answer one of these three questions. If something doesn't, it doesn't belong on the dashboard.

---

## Core Measures

```dax
Average Price =
AVERAGE ( products[price] )
```

```dax
Total Products =
COUNTROWS ( products )
```

```dax
Average Rating =
AVERAGE ( products[rating] )
```

```dax
Above-Category-Average Flag =
VAR CategoryAvg =
    CALCULATE (
        AVERAGE ( products[price] ),
        ALLEXCEPT ( products, products[category] )
    )
RETURN
    IF ( products[price] > CategoryAvg, "Above Avg", "At/Below Avg" )
```
> Answers Question 1. `ALLEXCEPT` keeps the category filter but removes every other filter, so this recalculates correctly per-row regardless of what else is sliced.

```dax
Weighted Avg Rating =
AVERAGEX ( products, products[rating] * products[review_count] ) /
SUMX ( products, products[review_count] )
```
> A simple `AVERAGE(rating)` treats a product with 3 reviews the same as one with 4,000. This weights by review volume \u2014 answers Question 2 more honestly than a flat average.

---

## Interactive / Filtered Measures

```dax
Filtered Avg Price =
CALCULATE (
    AVERAGE ( products[price] ),
    products[price] >= MIN ( price_range_slicer[value] ),
    products[price] <= MAX ( price_range_slicer[value] )
)
```
> Answers Question 3. Requires a `price_range_slicer` table (see README \u2014 "Adding the price-range slicer"). This measure automatically recalculates as the viewer drags the slicer; that's what makes the dashboard interactive rather than a set of static charts.

```dax
Rating Gap vs Category =
VAR CategoryAvgRating =
    CALCULATE (
        AVERAGE ( products[rating] ),
        ALLEXCEPT ( products, products[category] )
    )
RETURN
    products[rating] - CategoryAvgRating
```
> Directly surfaces Question 2: which products are dragging satisfaction down relative to their own category, not the whole catalog.

---

## Suggested Visuals \u2192 Measures Mapping

| Visual | Measures used | Answers |
|---|---|---|
| KPI card: Avg Price | `Average Price` | context for all 3 |
| KPI card: Avg Rating | `Average Rating` or `Weighted Avg Rating` | Q2 |
| KPI card: Total Products | `Total Products` | context |
| Bar chart: price by category/brand | `Average Price`, `Above-Category-Average Flag` (as a legend/color field) | Q1 |
| Scatter: price vs. rating | `products[price]`, `Weighted Avg Rating` | Q2 |
| Bar chart: rating gap by product | `Rating Gap vs Category` | Q2 |
| Any visual + price-range slicer | `Filtered Avg Price` | Q3 |

---

## Build Order (don't skip this order \u2014 it's the point of the session)

1. Write the 3-questions brief above, on paper, before opening Power BI.
2. Connect to data, shape tables (see README).
3. Build the 3 KPI cards first (`Average Price`, `Average Rating`, `Total Products`).
4. Build the category/brand comparison bar chart, add `Above-Category-Average Flag`.
5. Add the scatter (price vs. rating) and the rating-gap bar chart.
6. Add the price-range slicer **last**, wire up `Filtered Avg Price`.
7. Publish, then review the finished dashboard against the original 3 questions \u2014 cut anything that doesn't answer one.
