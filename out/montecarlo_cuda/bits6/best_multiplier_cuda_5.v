
`timescale 1ns/1ps
module multiplier (
    input [5:0] A,
    input [5:0] B,
    output [11:0] P);

    // Generación de productos parciales (sin redundancias)
    wire pp0 = (~A[0]&~B[5]);
    wire pp1 = (~A[3]&~B[5]);
    wire pp2 = (A[4]&B[4]);
    wire pp3 = (A[2]&B[5]);
    wire pp4 = (A[1]&~B[1]);
    wire pp5 = (A[5]&B[3]);
    wire pp6 = (~A[3]&~B[1]);
    wire pp7 = (A[5]&~B[5]);
    wire pp8 = (A[5]&B[1]);
    wire pp9 = (~A[2]&~B[5]);
    wire pp10 = (~A[0]&B[5]);
    wire pp11 = (~A[1]&B[3]);
    wire pp12 = (A[5]&B[4]);
    wire pp13 = (A[5]&~B[4]);
    wire pp14 = (~A[1]&B[0]);
    wire pp15 = (~A[0]&B[0]);
    wire pp16 = (~A[2]&~B[0]);
    wire pp17 = (A[3]&~B[0]);
    wire pp18 = (~A[5]&B[4]);
    wire pp19 = (~A[4]&B[0]);
    wire pp20 = (A[3]&B[3]);
    wire pp21 = (A[2]&~B[3]);
    wire pp22 = (A[0]&~B[5]);
    wire pp23 = (A[5]&B[2]);
    wire pp24 = (A[5]&~B[2]);
    wire pp25 = (A[2]&~B[5]);
    wire pp26 = (~A[1]&~B[2]);
    wire pp27 = (A[1]&~B[4]);
    wire pp28 = (~A[2]&~B[1]);
    wire pp29 = (A[3]&~B[3]);
    wire pp30 = (A[3]&B[2]);
    wire pp31 = (~A[0]&~B[1]);
    wire pp32 = (~A[0]&~B[3]);
    wire pp33 = 0;
    wire pp34 = (A[4]&B[1]);
    wire pp35 = (~A[5]&B[5]);
    wire pp36 = (A[4]&~B[1]);
    wire pp37 = (~A[5]&~B[0]);
    wire pp38 = (A[2]&~B[4]);
    wire pp39 = (~A[4]&B[1]);
    wire pp40 = (~A[4]&~B[5]);
    wire pp41 = (~A[2]&B[3]);
    wire pp42 = (~A[4]&B[3]);
    wire pp43 = (A[0]&B[0]);
    wire pp44 = (~A[1]&B[1]);
    wire pp45 = (~A[3]&~B[0]);
    wire pp46 = (A[3]&~B[5]);
    wire pp47 = (A[4]&~B[4]);
    wire pp48 = (~A[1]&~B[5]);
    wire pp49 = (A[1]&B[5]);
    wire pp50 = (A[0]&~B[1]);
    wire pp51 = (~A[3]&B[4]);
    wire pp52 = (A[1]&B[3]);

    // Suma de productos parciales
    wire [5:0] columna12 = pp0 + pp7 + pp22 + pp33 + pp46;
    wire [5:0] columna11 = pp1 + pp12 + pp23 + pp7;
    wire [5:0] columna10 = pp2 + pp24 + pp1 + pp33 + pp47;
    wire [5:0] columna9 = pp3 + pp13 + pp25 + pp34 + pp40 + pp48;
    wire [5:0] columna8 = pp4 + pp14 + pp26 + pp35 + pp6 + pp49;
    wire [5:0] columna7 = pp5 + pp15 + pp27 + pp36 + pp41 + pp16;
    wire [5:0] columna6 = pp6 + pp16 + pp9 + pp37 + pp42 + pp50;
    wire [5:0] columna5 = pp7 + pp17 + pp28 + pp3 + pp43 + pp40;
    wire [5:0] columna4 = pp8 + pp18 + pp29 + pp28 + pp44 + pp32;
    wire [5:0] columna3 = pp9 + pp19 + pp30 + pp38 + pp45 + pp41;
    wire [5:0] columna2 = pp10 + pp20 + pp31 + pp39 + pp17 + pp51;
    wire [5:0] columna1 = pp11 + pp21 + pp32 + pp2 + pp34 + pp52;
    assign P = (columna12 << 11) + (columna11 << 10) + (columna10 << 9) + (columna9 << 8) + (columna8 << 7) + (columna7 << 6) + (columna6 << 5) + (columna5 << 4) + (columna4 << 3) + (columna3 << 2) + (columna2 << 1) + (columna1 << 0);

endmodule