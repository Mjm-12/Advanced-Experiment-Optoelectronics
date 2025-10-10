.model 2SK879-Y NJF(Vto=-1.4 Beta=1.1m Lambda=1m Is=10f Isr=100f Vk=100 Alpha=10u Rd=30 Rs=30 CGD=9p CGS=9p Pb=0.3 M=0.38 mfg=TOSHIBA)
.model 2SK879-GR NJF(Vto=-2.4 Beta=0.77m Lambda=1m Is=10f Isr=100f Vk=100 Alpha=10u Rd=30 Rs=30 CGD=9p CGS=9p Pb=0.3 M=0.38 mfg=TOSHIBA)
.model 2SK880-GR NJF(Vto=-0.42 Beta=30m Lambda=5m Is=10f Isr=100f Vk=100 Alpha=10u Rd=15 Rs=15 CGD=9p CGS=15p Pb=0.55 M=0.33 mfg=TOSHIBA)
.model 2SK880-BL NJF(Vto=-0.75 Beta=23m Lambda=5m Is=10f Isr=100f Vk=100 Alpha=10u Rd=15 Rs=15 CGD=9p CGS=15p Pb=0.55 M=0.33 mfg=TOSHIBA)

* 879: C をデータシート準拠に
.model 2SK879-Y  NJF(Vto=-1.4  Beta=1.1m  Lambda=1m  Is=10f Isr=100f Vk=100 Alpha=10u  Rd=30 Rs=30  CGD=2.6p CGS=5.6p  Pb=0.55 M=0.33 mfg=TOSHIBA)
.model 2SK879-GR NJF(Vto=-2.4  Beta=0.77m Lambda=1m  Is=10f Isr=100f Vk=100 Alpha=10u  Rd=30 Rs=30  CGD=2.6p CGS=5.6p  Pb=0.55 M=0.33 mfg=TOSHIBA)

* 880: C を是正、gm を typ へ寄せる
.model 2SK880-GR NJF(Vto=-0.42 Beta=18m  Lambda=5m  Is=10f Isr=100f Vk=100 Alpha=10u  Rd=15 Rs=15  CGD=3p   CGS=10p  Pb=0.55 M=0.33 mfg=TOSHIBA)
.model 2SK880-BL NJF(Vto=-0.75 Beta=12m  Lambda=5m  Is=10f Isr=100f Vk=100 Alpha=10u  Rd=15 Rs=15  CGD=3p   CGS=10p  Pb=0.55 M=0.33 mfg=TOSHIBA)