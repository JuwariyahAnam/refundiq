"""Creates the mock e-commerce database: customers, orders, shipments, past tickets.
Run: python seed.py   (safe to re-run; it resets everything)"""
import sqlite3
from config import DB_PATH

CUSTOMERS = [
    ("C001", "Priya Sharma"), ("C002", "Rahul Verma"), ("C003", "Aisha Khan"),
    ("C004", "Vikram Rao"), ("C005", "Sneha Iyer"), ("C006", "Arjun Mehta"),
    ("C007", "Meera Nair"), ("C008", "Karan Singh"), ("C009", "Fatima Sheikh"),
    ("C010", "Dev Patel"), ("C011", "Rohan Gupta"), ("C012", "Neha Joshi"),
    ("C013", "Sameer Ali"), ("C014", "Kavya Reddy"), ("C015", "Imran Qureshi"),
    ("C016", "Zoya Siddiqui"), ("C017", "Aditya Kulkarni"), ("C018", "Riya Desai"),
]

# order_id, customer, item, category, price, order_date, serial, details, tracking_id
ORDERS = [
    ("O1001", "C001", "Silicone phone case", "accessories", 499, "2026-10-21", None, "", "TRK1001"),
    ("O1002", "C002", "Cotton T-shirt (size M)", "apparel", 899, "2026-10-15", None, "", "TRK1002"),
    ("O1003", "C003", "Glass electric kettle", "home_appliance", 1499, "2026-10-24", None, "", "TRK1003"),
    ("O1004", "C005", "Wireless ANC headphones", "electronics", 8999, "2026-10-22", "HP-77821", "", "TRK1004"),
    ("O1005", "C006", "Smartwatch Pro", "electronics", 12499, "2026-10-14", "SW-55310", "", "TRK1005"),
    ("O1006", "C007", "Bluetooth speaker", "electronics", 3999, "2026-10-23", "BS-11092",
     "Box contents: speaker unit, USB-C charging cable (listed value Rs 299), manual", "TRK1006"),
    ("O1007", "C008", "Mixer grinder 750W", "home_appliance", 1800, "2026-10-17", None, "", "TRK1007"),
    ("O1008", "C009", "Ceramic table lamp", "home_decor", 1299, "2026-10-25", None, "", "TRK1008"),
    ("O1009", "C010", "Running shoes (UK 9, blue)", "footwear", 1899, "2026-10-22", None, "", "TRK1009"),
    ("O1010", "C011", "Slim-fit jeans", "apparel", 1199, "2026-09-05", None, "", "TRK1010"),
    ("O1011", "C012", "Innerwear pack of 3", "innerwear", 599, "2026-10-22", None, "", "TRK1011"),
    ("O1012", "C004", "True wireless earbuds", "electronics", 1499, "2026-10-23", "EB-30017", "", "TRK1012"),
    ("O1013", "C013", "Power bank 20000mAh", "electronics", 1799, "2026-10-19", "PB-44120", "", "TRK1013"),
    ("O1015", "C015", "Yoga mat", "fitness", 999, "2026-10-12", None, "", "TRK1015"),
    ("O1016", "C016", "Winter jacket", "apparel", 1999, "2026-08-27", None, "", "TRK1016"),
    ("O1017", "C017", "Ergonomic office chair", "furniture", 6499, "2026-10-26", None, "", "TRK1017"),
    ("O1018", "C018", "Steel water bottle", "kitchen", 699, "2026-10-20", None, "", "TRK1018"),
]

# tracking_id, status, expected_date, delivered_date, simulate
SHIPMENTS = [
    ("TRK1001", "DELIVERED", "2026-10-25", "2026-10-25", None),
    ("TRK1002", "DELIVERED", "2026-10-20", "2026-10-20", None),
    ("TRK1003", "DELIVERED", "2026-10-28", "2026-10-28", None),
    ("TRK1004", "DELIVERED", "2026-10-26", "2026-10-26", None),
    ("TRK1005", "DELIVERED", "2026-10-18", "2026-10-18", None),
    ("TRK1006", "DELIVERED", "2026-10-27", "2026-10-27", None),
    ("TRK1007", "DELIVERED", "2026-10-21", "2026-10-21", None),
    ("TRK1008", "DELIVERED", "2026-10-29", "2026-10-29", None),
    ("TRK1009", "DELIVERED", "2026-10-26", "2026-10-26", None),
    ("TRK1010", "DELIVERED", "2026-09-10", "2026-09-10", None),
    ("TRK1011", "DELIVERED", "2026-10-26", "2026-10-26", None),
    ("TRK1012", "DELIVERED", "2026-10-27", "2026-10-27", None),
    ("TRK1013", "DELIVERED", "2026-10-23", "2026-10-23", None),
    ("TRK1015", "IN_TRANSIT", "2026-10-17", None, "timeout"),  # courier API times out
    ("TRK1016", "DELIVERED", "2026-09-01", "2026-09-01", None),
    ("TRK1017", "IN_TRANSIT", "2026-11-02", None, None),
    ("TRK1018", "DELIVERED", "2026-10-25", "2026-10-25", None),
]

# Past tickets = the agent's long-term memory of each customer
PAST_TICKETS = [
    ("PT-850", "C001", "O0850", "2026-04-12", "Case colour faded", "APPROVE", 349, "AUTO_REFUNDED"),
    ("PT-870", "C003", "O0870", "2026-07-01", "Wrong colour bedsheet", "APPROVE", 799, "AUTO_REFUNDED"),
    ("PT-901", "C004", "O0901", "2026-09-08", "Headphones not working", "APPROVE", 1299, "AUTO_REFUNDED"),
    ("PT-902", "C004", "O0902", "2026-09-30", "Shoes wrong size", "APPROVE", 2199, "HUMAN_APPROVED"),
    ("PT-903", "C004", "O0903", "2026-10-15", "Item arrived damaged", "APPROVE", 899, "AUTO_REFUNDED"),
    ("PT-950", "C018", "O1018", "2026-10-28", "Bottle lid leaking", "APPROVE", 699, "AUTO_REFUNDED"),
]

# Refund ledger: one row per order (UNIQUE) makes refunds idempotent -> no duplicates
PAST_REFUNDS = [(f"RF-{t[0][3:]}", t[2], t[0], t[6], t[3] + " 12:00:00", "auto" if t[7] == "AUTO_REFUNDED" else "human")
                for t in PAST_TICKETS]

SCHEMA = """
DROP TABLE IF EXISTS customers; DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS shipments; DROP TABLE IF EXISTS tickets;
DROP TABLE IF EXISTS proof_requests; DROP TABLE IF EXISTS refunds;
CREATE TABLE customers (customer_id TEXT PRIMARY KEY, name TEXT);
CREATE TABLE orders (order_id TEXT PRIMARY KEY, customer_id TEXT, item TEXT, category TEXT,
  price REAL, order_date TEXT, serial_number TEXT, details TEXT, tracking_id TEXT);
CREATE TABLE shipments (tracking_id TEXT PRIMARY KEY, status TEXT, expected_date TEXT,
  delivered_date TEXT, simulate TEXT);
CREATE TABLE tickets (ticket_id TEXT PRIMARY KEY, customer_id TEXT, order_id TEXT,
  created_date TEXT, message TEXT, decision TEXT, refund_amount REAL, status TEXT,
  clause_ids TEXT, reasoning TEXT, flags TEXT, reviewer_note TEXT, reviewed_at TEXT);
CREATE TABLE refunds (refund_id TEXT PRIMARY KEY, order_id TEXT UNIQUE, ticket_id TEXT,
  amount REAL, created_at TEXT, approved_by TEXT);
CREATE TABLE proof_requests (id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id TEXT,
  proof_type TEXT, created_date TEXT);
"""


def reset_db():
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO customers VALUES (?,?)", CUSTOMERS)
    con.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?)", ORDERS)
    con.executemany("INSERT INTO shipments VALUES (?,?,?,?,?)", SHIPMENTS)
    con.executemany(
        "INSERT INTO tickets (ticket_id, customer_id, order_id, created_date, message, decision,"
        " refund_amount, status) VALUES (?,?,?,?,?,?,?,?)", PAST_TICKETS)
    con.executemany("INSERT INTO refunds VALUES (?,?,?,?,?,?)", PAST_REFUNDS)
    con.commit()
    con.close()


if __name__ == "__main__":
    reset_db()
    print(f"Database ready at {DB_PATH}")
