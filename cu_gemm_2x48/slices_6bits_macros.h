// 6 bits slice with 3 ops - bit sets  
#define S6_OP3_SET_1 (0x80)
#define S6_OP3_SET_2 (0x40)
#define S6_OP3_SET_3 (0x20)

// 6 bits slice masks 
#define S6_MASK1 (0xFC)
#define S6_MASK2 (0x7E)
#define S6_MASK3 (0x3F)

// adjacent bit of 4 bits slice for rounding
#define S6_OP3_RND_BIT_1 (0x2)
#define S6_OP3_RND_BIT_2 (0x1)

// bits for rounding condition
#define S6_OP3_IS_RND_1 (0x6)
#define S6_OP3_IS_RND_2 (0x3)

// overflow condition bits
#define S6_OP3_OVRFLW_1 (0x100)
#define S6_OP3_OVRFLW_2 (0x80)
