// 4 bits slice with 2 ops - bit sets
#define S4_OP2_SET_1 (0xF0)
#define S4_OP2_SET_2 (0x0F)

// 4 bits slice with 3 ops - bit sets
#define S4_OP3_SET_1 (0xC0)
#define S4_OP3_SET_2 (0x30)
#define S4_OP3_SET_3 (0x0F)

// 4 bits slice with 5 ops - bit sets
#define S4_OP5_SET_1 (0x80)
#define S4_OP5_SET_2 (0x40)
#define S4_OP5_SET_3 (0x20)
#define S4_OP5_SET_4 (0x10)
#define S4_OP5_SET_5 (0x0F)

// 4 bits slice masks
#define S4_MASK1 (0xF0)
#define S4_MASK2 (0x78)
#define S4_MASK3 (0x3C)
#define S4_MASK4 (0x1E)
#define S4_MASK5 (0x0F) 

// adjacent bit of 4 bits slice for rounding
#define S4_OP2_RND_BIT_1 (0x8)

#define S4_OP3_RND_BIT_1 (0x8)
#define S4_OP3_RND_BIT_2 (0x2)

#define S4_OP5_RND_BIT_1 (0x8)
#define S4_OP5_RND_BIT_2 (0x4)
#define S4_OP5_RND_BIT_3 (0x2)
#define S4_OP5_RND_BIT_4 (0x1)

// bits for rounding condition
#define S4_OP2_IS_RND_1 (0x18)

#define S4_OP3_IS_RND_1 (0x18)
#define S4_OP3_IS_RND_2 (0x6)

#define S4_OP5_IS_RND_1 (0x18) 
#define S4_OP5_IS_RND_2 (0xC) 
#define S4_OP5_IS_RND_3 (0x6)
#define S4_OP5_IS_RND_4 (0x3)

// overflow condition bits
#define S4_OP2_OVRFLW_1 (0x100)

#define S4_OP3_OVRFLW_1 (0x100)
#define S4_OP3_OVRFLW_2 (0x40)

#define S4_OP5_OVRFLW_1 (0x100)
#define S4_OP5_OVRFLW_2 (0x80)
#define S4_OP5_OVRFLW_3 (0x40)
#define S4_OP5_OVRFLW_4 (0x20)
