# Key Performance Indicator (KPI) Definitions

## Revenue Metrics

### Total Revenue
**Definition**: Sum of (UnitPrice × Quantity × (1 - Discount)) across all OrderDetails.

**SQL Formula**:
```sql
SELECT SUM(UnitPrice * Quantity * (1 - Discount)) as TotalRevenue
FROM OrderDetails
```

### Average Order Value (AOV)
**Definition**: Total revenue divided by number of orders.

**Formula**: AOV = Total Revenue / Number of Orders

**Target**: $150+ per order

### Revenue Per Customer
**Definition**: Total revenue divided by unique customer count.

**Target**: $500+ per customer annually

## Order Metrics

### Order Count
**Definition**: Total number of orders in a given period.

### Order Fulfillment Rate
**Definition**: Percentage of orders successfully shipped.

**Target**: 95%+ fulfillment rate

### Average Items Per Order
**Definition**: Total quantity of items divided by number of orders.

**Target**: 5+ items per order

## Product Metrics

### Product Sales Velocity
**Definition**: Units sold per product per day.

**Formula**: Total Units Sold / Days in Period

### Inventory Turnover
**Definition**: Cost of goods sold divided by average inventory.

**Target**: 8+ turns per year

### Product Margin
**Definition**: (Selling Price - Cost) / Selling Price

**Target**: 30%+ margin on most products

## Customer Metrics

### Customer Lifetime Value (CLV)
**Definition**: Average revenue per customer multiplied by average customer lifespan.

**Target**: $2000+ CLV

### Repeat Customer Rate
**Definition**: Percentage of customers who made more than one purchase.

**Target**: 40%+ repeat rate

### Customer Acquisition Cost (CAC)
**Definition**: Total marketing spend divided by new customers acquired.

**Target**: CAC should be less than 1/3 of CLV

## Freight and Logistics

### Average Freight Cost
**Definition**: Sum of freight charges divided by number of orders.

**Target**: Keep below 5% of order value

### Shipping Cost Ratio
**Definition**: Total freight cost / Total revenue

**Target**: Below 8%
