from hope.hope_protocol import HopeConfig, HopeProtocol

hp = HopeProtocol(
    HopeConfig(
        p_hope=0.2,
        min_multiplier=2.0,
        max_multiplier=3.0,
        mode="symmetric",
    )
)

for i in range(10):
    r = 1.0
    print(f"raw={r}, transformed={hp.transform(r)}")
