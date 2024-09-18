from dash import Dash, html, dcc, Input, Output, callback, State
from concurrent.futures import ThreadPoolExecutor, as_completed
import dash_bootstrap_components as dbc
from datetime import datetime, date
import plotly.graph_objects as go
from dotenv import load_dotenv
import dash_ag_grid as dag
import polars as pl
import numpy as np
import requests
import pyarrow
import json
import os

load_dotenv()
access_key = os.getenv("URA_ACCESS_KEY")

base_url = "https://www.ura.gov.sg/uraDataService/" 
token_url = 'insertNewToken.action'
header = {'AccessKey': access_key, 'User-Agent': 'curl/7.68.0'}

# Get Token
token = requests.get(base_url + token_url, headers=header).json().get("Result")

# Add token for API requests
header.update({'Token':token})
pp_url = "invokeUraDS?service=PMI_Resi_Transaction&batch="

def fetch_ura_data(period):
    response = requests.get(base_url + pp_url + period, headers=header)
    if response.status_code == 200:
        return pl.DataFrame(response.json().get("Result"))
    else:
        return pl.DataFrame()  # Return empty DataFrame on error

# Use ThreadPoolExecutor to fetch data in parallel
# partitions = ['1', '2', '3', '4']
partitions = ['1']
pdf = pl.DataFrame()
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(
        fetch_ura_data, period): period for period in partitions}

    for future in as_completed(futures):
        tmp = future.result()
        pdf = pl.concat([pdf, tmp], how="diagonal_relaxed")

print("Completed data extraction from URA")

cols_select = ['month', 'street - project', 'floor', 'sales', 'district', 
               'region', 'mgmt', 'property', 'freehold', 'lease_left',
               'price', 'price_sqm', 'price_sqft', 'area_sqft', 'area_sqm']

yr, mth = datetime.now().year, datetime.now().month
selected_mths = pl.date_range(
    date(2024, 1, 1), date(yr, mth, 1), "1mo", eager=True).to_list()
selected_mths = [i.strftime("%Y-%m") for i in selected_mths[-int(6):]]

pdf = (pdf.explode('transaction').unnest("transaction").drop("x", "y"))
pdf = pdf.with_columns(
    pl.col("area").cast(pl.Decimal),
    pl.col('typeOfSale').cast(pl.Categorical).alias("sale_type"),
    (pl.col("contractDate").str.slice(0, 2) + 
        "-20" + pl.col("contractDate").str.slice(-2)).alias("date"),
    pl.when(pl.col("nettPrice").is_null())
      .then(pl.col("price"))
      .otherwise(pl.col("nettPrice")).cast(pl.Decimal).alias("new_price"),
    pl.int_ranges("noOfUnits").alias("list")
).explode('list').drop("price")

pdf = pdf.with_columns(
    (pl.col("new_price") / pl.col("noOfUnits")).cast(pl.Float64).alias("price"),
    (pl.col("area") / pl.col("noOfUnits")).cast(pl.Float64).alias("area_sqm"), 
    (pl.when(pl.col("tenure") != "Freehold")
     .then(pl.lit("No"))
     .otherwise(pl.lit("FH")) # keep original value
     .alias('freehold'))
)

# Calculations for lease
pdf = pdf.with_columns(
    pl.col("tenure")
    .str.extract_all(r"\d+")
    .list.to_struct(fields=["lease", "lease_start"])
    .alias("lease_info")
).unnest("lease_info").with_columns(
    pl.col("lease").cast(pl.Decimal),
    pl.col("lease_start").cast(pl.Decimal)
)

# A set of calculations for lease_left
pdf = pdf.with_columns(
    pl.col("date").str.split("-")
    .list.to_struct(fields=["month", "year"])
    .alias("sold_date")
).unnest("sold_date").with_columns(
    ((pl.col("year").cast(pl.Float32)) - pl.col("lease_start")).alias("lease_used")
).with_columns((pl.col("lease") - pl.col("lease_used")).alias("lease_left")
).with_columns(
    # Fill lease of sales where freehold is null and no lease is defined yet
    (
        pl.when(
            (pl.col("freehold").is_null())
            & (pl.col("lease_start").is_null())
            & (pl.col('lease_left').is_null())
        )
        .then(99)
        .otherwise(pl.col("lease_left"))
    ).alias("final_lease_left")
)

# Calculate more price and area metircs
pdf = pdf.with_columns(
    (pl.col("street") + " - " + pl.col("project")).alias("street - project"),
    (pl.col("price") / pl.col("area_sqm")).cast(pl.Float64).alias("price_sqm"),
    (pl.col("area_sqm") * 10.76391042).cast(pl.Float64).alias("area_sqft"),
).with_columns(
    (pl.col("price") / pl.col("area_sqft")).cast(pl.Float64).alias("price_sqft"),
    pl.col("date").str.to_date("%m-%Y").cast(pl.String).str.slice(0, 7).alias('month'),
    (pl.when(pl.col("noOfUnits") == "1")
     .then(pl.col("noOfUnits"))
     .otherwise(pl.lit(">1")) # keep original value
     .alias('sales')
    ),
    pl.col("propertyType").str.replace("Condominium", "Condo")
    .replace("Strata", "S.")
    .replace('Strata Semi-detached', "Semi-D")
    .replace('Strata Terrace', "Terrace")
    .replace('Strata Detached', "Detached")
    .replace("Executive Condo", "EC")
    .replace("Semi-detached", "Semi-D")
).drop('lease_left').rename({
    "propertyType": 'property',
    'marketSegment': "region",
    "floorRange": 'floor',
    "typeOfArea": "mgmt",
    "final_lease_left": 'lease_left'
}).select(cols_select).filter(
    pl.col("month").is_in(selected_mths)
)

# Initalise App
app = Dash(__name__, external_stylesheets=[
    {'src': 'https://cdn.tailwindcss.com'}, dbc.themes.BOOTSTRAP],
    requests_pathname_prefix="/private_housing/",
)
# From here on, I should be using the revamped code from public_housing.py
districts = pdf.select(pl.col("district")).unique().to_series().to_list()
districts.sort()
districts = ['All', ] + districts

regions = pdf.select(pl.col("region")).unique().to_series().to_list()
regions.sort()
regions = ['All', ] + regions

property = pdf.select(pl.col("property")).unique().to_series().to_list()
property.sort()

props = []
for prop in props:
    props.append({
        "label": html.Span([prop], style={
            'background-color': "#FFC0BD", 
            'border': "#FFC0BD",
            'color': 'black'}),
        "value": prop, "search": prop 
    })
#
# # Set initial min-max parameters
p_price_max = pdf.select("price").max().rows()[0][0]
p_price_min = pdf.select("price").min().rows()[0][0]
p_area_max = round(pdf.select("area_sqft").max().rows()[0][0], 2)
p_area_min = round(pdf.select("area_sqft").min().rows()[0][0], 2)
p_legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=.5)
p_chart_width, p_chart_height = 680, 550

def p_convert_price_area(price_type, area_type):
    """ Convert user price for table ftilers & Plotly labels """ 

    if price_type != 'price':
        price_type = 'price_sqft' if area_type == 'area_sqft' else 'price_sqm'

    return price_type


def p_grid_format(table: pl.DataFrame):
    """ Add custom formatting to AGrid Table Outputs """
    output = [
        {"field": "month", "sortable": True, 'width': 120, 'maxWidth': 150},
        {"field": "street - project", "sortable": True, 'width': 500, 'maxWidth': 800},
        {"field": "property", "sortable": True, 'width': 150, 'maxWidth': 150},
        {"field": "region", "sortable": True, 'width': 100, 'maxWidth': 100},
        {"field": "district", "sortable": True, 'width': 110, 'maxWidth': 150},
        {"field": "floor", "sortable": True, 'width': 100, 'maxWidth': 120},
        {"field": "sales", "sortable": True, 'width': 80, 'maxWidth': 100},
        {"field": "mgmt", "sortable": True, 'width': 100, 'maxWidth': 150},
        {"field": "lease_left", "sortable": True, 'width': 100, 'maxWidth': 150},
        {
            "field": "price", "sortable": True, 'width': 170, 'maxWidth': 200, 
            "valueFormatter": {"function": 
                               "d3.format('($,.2f')(params.value)"},
        }
    ]
    for col in table.columns:
        if 'price_' in col:
            output.append({
                "field": col, "sortable": True, 'width': 150, 'maxWidth': 180,
                "valueFormatter": {"function":
                                   "d3.format('($,.2f')(params.value)"},
            })
        elif "area" in col:
            output.append({
                "field": col, "sortable": True, 'width': 130, 'maxWidth': 180,
                "valueFormatter": {"function":
                                   "d3.format('(,.2f')(params.value)"},
            })
    return output

def p_df_filter(month, district, region, property, area_type, max_area, min_area,
              price_type, freehold, max_price, min_price, min_lease, max_lease,
              street, selected_mths, data_json):
    """Filter Polars DataFrame for Viz, based on inputs"""
    pdf = pl.DataFrame(json.loads(data_json)).lazy()

    # Conditional flags
    flags = []

    if property != "All":
        flags.append(
            pl.when(pl.col("property").is_in(property))
            .then(True)
            .otherwise(False)
            .alias("property_flag")
        )

    if freehold == "FH":
        flags.append(
            pl.when(pl.col("freehold") == "FH")
            .then(True)
            .otherwise(False)
            .alias("freehold_flag")
        )
    elif freehold != "All":
        flags.append(
            pl.when(pl.col("freehold") == "FH")
            .then(False)
            .otherwise(True)
            .alias("freehold_flag")
        )

    if region != "All":
        flags.append(
            pl.when(pl.col("region") == region)
            .then(True)
            .otherwise(False)
            .alias("region_flag")
        )

    if district != "All":
        flags.append(
            pl.when(pl.col("district") == district)
            .then(True)
            .otherwise(False)
            .alias("district_flag")
        )

    if max_lease:
        flags.append(
            pl.when(pl.col("lease_left") >= int(max_lease))
            .then(True)
            .otherwise(False)
            .alias("max_lease_flag")
        )

    if min_lease:
        flags.append(
            pl.when(pl.col("lease_left") <= int(min_lease))
            .then(True)
            .otherwise(False)
            .alias("min_lease_flag")
        )

    if street:
        flags.append(
            pl.when(pl.col("street - project").str.contains(street.upper()))
            .then(True)
            .otherwise(False)
            .alias("street_flag")
        )

    if max_price:
        flags.append(
            pl.when(pl.col(price_type) <= max_price)
            .then(True)
            .otherwise(False)
            .alias("max_price_flag")
        )

    if min_price:
        flags.append(
            pl.when(pl.col(price_type) >= min_price)
            .then(True)
            .otherwise(False)
            .alias("min_price_flag")
        )

    if max_area:
        flags.append(
            pl.when(pl.col(area_type) <= max_area)
            .then(True)
            .otherwise(False)
            .alias("max_area_flag")
        )

    if min_area:
        flags.append(
            pl.when(pl.col(area_type) >= min_area)
            .then(True)
            .otherwise(False)
            .alias("min_area_flag")
        )

    # Filter and rounding for price and area columns
    price_type = p_convert_price_area(price_type, area_type)

    if area_type == 'area_sqft':
        rd_col = [
            pl.col('price').round(2),
            pl.col('price_sqft').round(2),
            pl.col('area_sqft').round(2)
        ]
        drop_columns = ["price_sqm", 'area_sqm']
    else:
        rd_col = [
            pl.col('price').round(2),
            pl.col('price_sqm').round(2),
            pl.col('area_sqm').round(2)
        ]
        drop_columns = ['price_sqft', "area_sqft"]

    return pdf.with_columns(rd_col + flags).drop(drop_columns).collect()

app.layout = html.Div([
    html.Div(
        id="p-data-store",
        style={"display": "none"},
        children=pdf.write_json(),
    ),
    dcc.Store(id='p-filtered-data'),
    html.H3(
        children="Private Home Transaction Search Tool",
        style={'font-weight': 'bold', 'font-size': '26px'},
        className="mb-4 pt-4 px-4",
    ),
    dcc.Markdown(
        """ Building from my public housing resale search tool, that I built, I
        am trying to do the same thing with the private housing resales data
        taken from URA. I realised that it wasn't as easy as just changing the
        data set and re-running the same set of code, as our private and public
        housing market has quite a few differences, from freehold-vs-leasehold,
        different housing types, to region and district zones being used in our
        private housing market.
        """, className="px-4"),
    dcc.Markdown("""
        I realised that the original charts I used to showcase some of the data
        patterns for our public housing markets may not be as useful for our
        private housing market, so I am still thinking how to work on that.
        """, className="px-4"), 
    dcc.Markdown("""
        Nonetheless, this is a Work In Progress, so please stay tune for more
        updates! Also, do reach out to me if you have any suggestions or
        feedback regaining the current work on the Singapore property market!
        """, className="px-4",
    ),
    dbc.Row([
        dbc.Col(
            dbc.Button(
                "Filters",
                id="p-collapse-button",
                className="mb-3",
                color="danger",
                n_clicks=0,
                style={"verticalAlign": "top"}
            ),
            width="auto"
        ),
        dbc.Col(
            dcc.Loading([
                html.P(
                    id="p-dynamic-text",
                    style={"textAlign": "center", "padding-top": "10px"}
                )], type="circle", color="rgb(220, 38, 38)"),
            width="auto"
        ),
        dbc.Col(
            dbc.Button(
                "Caveats",
                id="p-collapse-caveats",
                className="mb-3",
                color="danger",
                n_clicks=0,
                style={"verticalAlign": "top"}
            ),
            width="auto"
        ),
    ], justify="center"),
    dbc.Collapse(
        dbc.Card(
            dbc.CardBody([
                dcc.Markdown("""
                1. En-bloc and bulk private home sales are provided by URA with 
                their total sales and total units sold in one single record. The
                average sales of these records are calculated by taking their
                total sale prices divideded by the number of units sold for that
                transaction.
                2. This is still a very much work in progress tool.
                """)
            ], style={"textAlign": "left", "color": "#555", "padding": "5px"}),
        ),
        id="p-caveats",
        is_open=False,
    ),
    dbc.Collapse(
        dbc.Card(dbc.CardBody([
            html.Div([
                html.Div([
                    html.Label("Months"),
                    dcc.Dropdown(options=[3, 6], value=6, id="p_month")
                ], style={"display": "inline-block",
                          "width": "7%", "padding": "10px"},
                ),
                html.Div([
                    html.Label("Region"),
                    html.Div(dcc.Dropdown(
                        options=regions, value="All", id="p_region")),
                ], style={"display": "inline-block",
                          "width": "9%", "padding": "10px"},
                ),
                html.Div([
                    html.Label("District"),
                    html.Div(dcc.Dropdown(
                        options=districts, value="All", id="p_district")),
                ], style={"display": "inline-block",
                          "width": "7%", "padding": "10px"},
                ),
                html.Div([
                    html.Label("Property"),
                    dcc.Dropdown(multi=True, options=props,
                                 value=property, id="p_property"),
                ], style={"display": "inline-block",
                          "width": "40%", "padding": "10px"},
                ),
                html.Div([
                    html.Label("Lease Type"),
                    dcc.Dropdown(options=[
                        {"label": "Freehold", 'value': "FH"},
                        {"label": "Leasehold", 'value': "LH"},
                        {"label": "All", 'value': "All"}
                    ], value='All', id="p_freehold"),
                ], style={"display": "inline-block",
                          "width": "11%", "padding": "10px"},
                ),
                html.Div([
                    html.Label("Min Lease [Yrs]"),
                    dcc.Input(type="number",
                              placeholder="Add No.",
                              style={"display": "flex",
                                     "border-color": "#E5E4E2",
                                     "padding": "5px"},
                              id="p_min_lease"),
                ], style={"display": "flex",  "flexDirection": "column",
                          "width": "10%", "padding": "10px",
                          "verticalAlign": "top"}),
                html.Div([
                    html.Label("Max Lease [Yrs]"),
                    dcc.Input(type="number",
                              placeholder="Add No.",
                              style={"display": "flex",
                                     "border-color": "#E5E4E2",
                                     "padding": "5px"},
                              id="p_max_lease"),
                ], style={"display": "flex",  "flexDirection": "column",
                          "width": "10%", "padding": "10px",
                          "verticalAlign": "top"},
                ),
            ], style={"display": "flex", "flexDirection": "row",
                      "alignItems": "center"}
            ),
            # Area inputs
            html.Div([
                html.Div([
                    html.Label("Sq Feet | Sq M"),
                    html.Div(dcc.Dropdown(options=[
                        {'label': 'Sq Feet', 'value': 'area_sqft'},
                        {'label': 'Sq M', 'value': 'area_sqm'},
                        ], value="area_sqft", id="p_area_type")),
                ], style={"display": "flex", "flexDirection": "column",
                          "width": "12%", "padding": "10px"},
                ),
                html.Div([
                    html.Label("Min Area"),
                    dcc.Input(type="number",
                              placeholder="Add No.",
                              style={"display": "inline-block",
                                     "border-color": "#E5E4E2",
                                     "padding": "5px"},
                              id="p_min_area"),
                ], style={"display": "flex",  "flexDirection": "column",
                          "width": "12%", "padding": "5px"},
                ),
                html.Div([
                    html.Label("Max Area"),
                    dcc.Input(type="number",
                              placeholder="Add No.",
                              style={"display": "inline-block",
                                     "border-color": "#E5E4E2",
                                     "padding": "5px"},
                              id="p_max_area"),
                ], style={"display": "flex", "flexDirection": "column",
                          "width": "12%", "padding": "5px"},
                ),
                html.Div([
                    html.Label("Price | Price / Area"),
                    dcc.Dropdown(options=[
                        {"label": 'Price', "value": 'price'},
                        {"label": "Price / Area", "value": 'price_area'}
                    ], value="price", id="p_price_type"),
                ], style={"display": "flex", "flexDirection": "column",
                          "width": "14%", "padding": "5px"},
                ),
                html.Div([
                    html.Label("Min [ Price | Price / Area ]"),
                    dcc.Input(type="number",
                              placeholder="Add No.",
                              style={"display": "inline-block",
                                     "border-color": "#E5E4E2",
                                     "padding": "5px"},
                              id="p_min_price"),
                ], style={"display": "flex", "flexDirection": "column",
                          "width": "15%", "padding": "5px"},
                ),
                html.Div([
                    html.Label("Max [ Price | Price / Area ]"),
                    dcc.Input(type="number",
                              style={"display": "inline-block",
                                     "border-color": "#E5E4E2",
                                     "padding": "5px"},
                              placeholder="Add No.",
                              id="p_max_price"),
                ], style={"display": "flex", "flexDirection": "column",
                          "width": "15%", "padding": "5px"},
                ),
                html.Div([
                    html.Label("Submit", style={'margin-top': '12px'}),
                    dbc.Button('Submit', 
                               id='p-submit-button',
                               className="mb-3",
                               color="danger",
                               n_clicks=0,
                               style={"verticalAlign": "top"})
                ], style={"display": "flex", "flexDirection": "column",
                          "width": "8%", "padding": "5px"},
                )
            ], style={"display": "flex", "flexDirection": "row",
                      "alignItems": "center"}),
            html.Div([
                html.Label("""Search by Street - Project
        ( Add | separator to include >1 Street / Project )"""),
                dcc.Input(type="text",
                          style={"display": "inline-block",
                                 "border-color": "#E5E4E2",
                                 "padding": "5px"},
                          placeholder="Type the Street / Project here",
                          id="p_street"),
            ], style={"display": "flex", "flexDirection": "column",
                      "width": "45%", "padding": "5px"},
            ),
        ]
        )),
        id="p-collapse",
        is_open=True,
    ),
    # Text box to display dynamic content
    html.Div([
        html.Div([
            dcc.Loading([
                html.Div([
                    html.H3(
                        "Filtered Private Housing Transactions",
                        style={
                            "font-size": "20px",
                            "textAlign": "left",
                            "margin-top": "15px",
                            "margin-bottom": "5px",
                        },
                    ),
                    dag.AgGrid(
                        id="p-price-table",
                        columnDefs=p_grid_format(pdf),
                        rowData=pdf.to_dicts(),
                        className="ag-theme-balham",
                        columnSize="responsiveSizeToFit",
                        dashGridOptions={
                            "pagination": True,
                            "paginationAutoPageSize": True,
                        },
                    ),
                ], style={
                    "height": 450,
                    "width": 1200,
                    "display": "inline-block",
                },
                )], type="circle", color="rgb(220, 38, 38)"),
        ], style=dict(display="flex"),
        ),
        # dcc.Loading([
        #     html.Div([
        #         dcc.Graph(id="g0", style={"display": "inline-block", "width": "48%"}),
        #         dcc.Graph(id="g2", style={"display": "inline-block", "width": "38%"}),
        #     ], style={
        #         "display": "flex",
        #         "justify-content": "flex-start",
        #         "width": "100%"}
        #     )], type="circle", color="rgb(220, 38, 38)")
    ],
        style={"display": "flex",
               "flexDirection": "column",
               "justifyContent": "center",
               "alignItems": "center",
               "minHeight": "100vh",
               "textAlign": "center",
               }
    )
])


# Standardised Dash Input-Output states
private_basic_state = [
    State('p_district', 'value'),
    State('p_region', 'value'),
    State('p_area_type', 'value'),
    State('p_price_type', 'value'),
    State('p_freehold', 'value'),
    State('p_max_lease', 'value'),
    State('p_min_lease', 'value')
]
private_addon_state = [
    State('p_month', 'value'),
    State('p_property', 'value'),
    State('p_max_area', 'value'),
    State('p_min_area', 'value'),
    State('p_max_price', 'value'),
    State('p_min_price', 'value'),
    State('p_street', 'value'),
    State('p-data-store', 'children')
]
private_states = private_basic_state + private_addon_state

@callback(Output("p-filtered-data", "data"),
          Input('p-submit-button', 'n_clicks'), 
          private_states)
def p_filtered_data(n_clicks, district, region, area_type, price_type, freehold, 
                  max_lease, min_lease, month, property, max_area, min_area, 
                  max_price, min_price, street, data_json):

    return p_df_filter(month, district, region, property, area_type, max_area, 
                       min_area, price_type, freehold, max_price, min_price,
                       min_lease, max_lease, street, selected_mths, 
                       data_json).to_dicts()


@callback(Output("p-price-table", "rowData"),
          Output('p-price-table', 'columnDefs'),
          Input('p-filtered-data', 'data'),
          State('p_area_type', 'value'),
          State('p_price_type', 'value'))
def p_update_table(data, area_type, price_type):
    """ Table output to show all searched transactions """
    output = [{}]
    df = pl.DataFrame(data)
    if df is not None:
        df = df.with_columns((
                pl.when(pl.col('freehold') == "No")
                    .then(pl.col('lease_left').cast(pl.String))
                    .otherwise(pl.col('freehold')
                ).alias('lease'))
        ).drop('freehold', 'lease_left')
        flags = [i for i in df.columns if 'flag' in i]
        df = df.filter(~pl.any_horizontal(pl.col('*') == False)).drop(flags)

        output = df.to_dicts()
        columnDefs=p_grid_format(df)
    return output, columnDefs


@callback(Output("p-dynamic-text", "children"),
          Input('p-filtered-data', 'data'),
          private_basic_state)
def p_update_text(data, district, region, area_type, price_type, freehold, 
                max_lease, min_lease):
    """ Summary text for searched output """

    df = pl.DataFrame(data)
    df = df.with_columns((
            pl.when(pl.col('freehold') == "No")
                .then(pl.col('lease_left').cast(pl.String))
                .otherwise(pl.col('freehold')
            ).alias('lease'))
    ).drop('freehold', 'lease_left')
    flags = [i for i in df.columns if 'flag' in i]
    df = df.filter(~pl.any_horizontal(pl.col('*') == False)).drop(flags)

    records = df.shape[0]
    if records > 0:
        area_min = round(min(df.select(area_type).to_series()), 2)
        area_max = round(max(df.select(area_type).to_series()), 2)

        if price_type == 'price':
            price_min = min(df.select(price_type).to_series())
            price_max = max(df.select(price_type).to_series())
            price_label = 'price'

        else:
            price_type = "price_" + area_type.split('_')[-1]
            price_min = min(df.select(price_type).to_series())
            price_max = max(df.select(price_type).to_series())
            price_label = f"Price / {area_type.split('_')[-1]}"

        text = f"""<b>You searched : </b>
        <b>District</b>: {district} |
        <b>Region</b>: {region} |
        <b>Lease Type</b>: {freehold} |
        <b>{price_label}</b>: ${price_min:,} - ${price_max:,} |
        <b>{area_type}</b>: {area_min:,} - {area_max:,}
        """

        if min_lease and max_lease:
            text += f" | <b>Lease from</b> {min_lease:,} - {max_lease:,}"
        elif min_lease:
            text += f" | <b>Lease >= </b> {min_lease:,}"
        elif max_lease:
            text += f" | <b>Lease =< </b> {max_lease:,}"

        text += f" | <b>Total records</b>: {records:,}"
    else:
        text = "<b><< YOUR SEARCH HAS NO RESULTS >></b>"

    return dcc.Markdown(text, dangerously_allow_html=True)


# @callback(Output("g0", "figure"),
#           Input('p-filtered-data', 'data'),
#           private_basic_state)
# def update_g0(data, district, region, area_type, price_type, freehold, 
#               max_lease, min_lease):
#     """ Scatter Plot of Price to Price / Sq Area """
#     fig = go.Figure()
#     df = pl.DataFrame(data)
#
#     if df is not None:
#         df = df.filter(~pl.col('lease_left').is_null())
#         flags = [i for i in df.columns if 'flag' in i]
#         non_df = df.filter(pl.any_horizontal(pl.col('*') == False)).drop(flags)
#         df = df.filter(~pl.any_horizontal(pl.col('*') == False)).drop(flags)
#
#         price_type = convert_price_area(price_type, area_type)
#         price_label = 'price_sqm' if area_type == 'area_sqm' else 'price_sqft'
#
#         base_cols = ['lease_left', 'district', 'street - project', area_type]
#         customdata_set = list(df[base_cols].to_numpy())
#
#         fig.add_trace(
#             go.Scattergl(
#                 y=non_df.select('price').to_series(),  # unchanged
#                 x=non_df.select(price_label).to_series(),
#                 mode='markers',
#                 hoverinfo='skip',
#                 marker={"color": "#FFC0BD", "opacity": 0.5},
#                 name='Rest of SG'
#             ))
#         fig.add_trace(
#             go.Scattergl(
#                 y=df.select('price').to_series(),  # unchanged
#                 x=df.select(price_label).to_series(),
#                 customdata=customdata_set,
#                 hovertemplate='<i>Price:</i> %{y:$,}<br>' +
#                 '<i>Area:</i> %{customdata[3]:,}<br>' +
#                 '<i>Price/Area:</i> %{x:$,}<br>' +
#                 '<i>District :</i> %{customdata[1]}<br>' +
#                 '<i>Street / Project:</i> %{customdata[2]}<br>' +
#                 '<i>Lease Left:</i> %{customdata[0]}',
#                 mode='markers',
#                 marker={"color": "rgb(220, 38, 38)", "opacity": 0.9},
#                 name='Selected Data'
#             ))
#         fig.update_layout(
#             title="<b>Home Prices vs Price / Area<b>",
#             yaxis={"title": "price", "gridcolor" :'#d3d3d3', "showspikes": True},
#             xaxis={"title": f"{price_type}", "gridcolor":'#d3d3d3', "showspikes": True},
#             width=chart_width,
#             height=chart_height,
#             legend=legend,
#             plot_bgcolor='white',
#             margin=dict(l=5,r=5)
#         )
#     return fig
#
#
# @callback(Output("g2", "figure"),
#           Input('p-filtered-data', 'data'),
#           private_basic_state)
# def update_g2(data, district, region, area_type, price_type, freehold, 
#               max_lease, min_lease):
#     """ Price to Lease Left Plot """
#     fig = go.Figure()
#     df = pl.DataFrame(data)
#
#     if df is not None:
#         flags = [i for i in df.columns if 'flag' in i]
#         non_df = df.filter(pl.any_horizontal(pl.col('*') == False)).drop(flags)
#         df = df.filter(~pl.any_horizontal(pl.col('*') == False)).drop(flags)
#
#         # Transform user inputs into table usable columns
#         price_type = convert_price_area(price_type, area_type)
#
#         price_label = 'price_sqm' if area_type == 'area_sqm' else "price_sqft"
#         base_cols = ['price', price_label, 'district', 'street - project', area_type]
#         customdata_set = list(df[base_cols].to_numpy())
#
#         fig.add_trace(
#             go.Scattergl(
#                 y=non_df.select(price_type).to_series(),  # unchanged
#                 x=non_df.select("lease_left").to_series(),
#                 mode='markers',
#                 hoverinfo='skip',
#                 marker={"color": "#FFC0BD", "opacity": 0.5},
#                 name='Rest of SG'
#             ))
#
#         fig.add_trace(
#             go.Scattergl(
#                 y=df.select(price_type).to_series(),  # unchanged
#                 x=df.select('lease_left').to_series(),
#                 customdata=customdata_set,
#                 hovertemplate='<i>Price:</i> %{customdata[0]:$,}<br>' +
#                 '<i>Area:</i> %{customdata[4]:,}<br>' +
#                 '<i>Price/Area:</i> %{customdata[1]:$,}<br>' +
#                 '<i>District :</i> %{customdata[2]}<br>' +
#                 '<i>Street / Project:</i> %{customdata[3]}<br>' +
#                 '<i>Lease Left:</i> %{x}',
#                 mode='markers',
#                 marker={"color": "rgb(220, 38, 38)", "opacity": 0.9},
#                 name="Selected Data"
#             )
#         )
#         fig.update_layout(
#             title="<b>Home Prices vs Lease Left<b>",
#             yaxis={"title": f"{price_type}", 'gridcolor': '#d3d3d3', "showspikes": True},
#             xaxis={"title": "lease_left", 'gridcolor': '#d3d3d3', "showspikes": True},
#             width=chart_width,
#             height=chart_height,
#             legend=legend,
#             plot_bgcolor='white',
#             margin=dict(l=5,r=5)
#         )
#     return fig
#
#
# @callback(
#     Output("collapse", "is_open"),
#     [Input("collapse-button", "n_clicks")],
#     [State("collapse", "is_open")])
# def toggle_collapse(n, is_open):
#     return not is_open if n else is_open
#
# @callback(
#     Output("caveats", "is_open"),
#     [Input("collapse-caveats", "n_clicks")],
#     [State("caveats", "is_open")])
# def toggle_caveat(n, is_open):
#     return not is_open if n else is_open
#
#
if __name__ == "__main__":
    app.run(debug=True)
