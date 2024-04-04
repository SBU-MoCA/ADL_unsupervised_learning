# run this script on DMZ server 2 to export data from InfluxDB to csv.
# input the start and stop time range to dump data, in UTC time.

#nodes="b8-27-eb-a3-b3-2a"
nodes="\
b8-27-eb-4f-46-d7 \
b8-27-eb-ec-77-37 \
b8-27-eb-b4-f8-c2 \
b8-27-eb-4e-d2-eb \
b8-27-eb-02-d4-0b \
b8-27-eb-a3-b3-2a \
b8-27-eb-2c-3e-07 \
b8-27-eb-86-23-51 \
b8-27-eb-cf-59-2a \
b8-27-eb-85-a7-83 \
b8-27-eb-92-28-87 \
b8-27-eb-3f-d0-0b \
b8-27-eb-1b-02-d2 \
b8-27-eb-be-fd-bf \
b8-27-eb-63-ae-61 \
b8-27-eb-dc-a9-b5"

for node in $nodes; do
        echo $node
        # sudo curl -XPOST http://172.28.220.119:8086/api/v2/query?org=ece_moca -sS \
        sudo curl -XPOST localhost:8086/api/v2/query?org=ece_moca -sS \
        -H 'Accept: application/csv' \
        -H 'Content-type:application/vnd.flux' \
        -H 'Authorization: Token y7XJp76dUWV2naal65BS-OpPYDSVJqqOmKAU7kTzN14vmE0HnTVVpQZB58EuivEIBGkVHuxpHcOmGRgGb3Dtaw==' \
        -d 'from(bucket: "test")
          |> range(start: 2023-06-29T21:25:53Z, stop: 2023-06-29T21:37:35Z)
          |> filter(fn: (r) => r["_measurement"] == "UWB_data")
          |> filter(fn: (r) => r["_field"] == "imaginary" or r["_field"] == "real" or r["_field"] == "timestamp")
          |> filter(fn: (r) => r["node_ID"] == "'"$node"'")' > /data/ADL_data/$node.csv
done
