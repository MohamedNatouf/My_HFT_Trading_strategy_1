# Saxo FIX 4.4 client with session logon, symbology, Parties, and expanded order types.
import quickfix as fix
import quickfix44 as fix44
from typing import Optional, Dict, List

class FixClient(fix.Application):
    def __init__(self, settings: fix.SessionSettings, creds: Optional[Dict] = None, parties: Optional[List[Dict]] = None):
        super().__init__()
        self.settings = settings
        self.storeFactory = fix.FileStoreFactory(settings)
        self.logFactory = fix.FileLogFactory(settings)
        self.initiator: Optional[fix.SocketInitiator] = None
        self.sessions: List[fix.SessionID] = []
        self.creds = creds or {}
        self.parties = parties or []  # [{role: 3/11/38, id:'', src:'D'}]

    # FIX application callbacks
    def onCreate(self, sessionID):
        self.sessions.append(sessionID)

    def onLogon(self, sessionID):
        print(f"FIX logon: {sessionID}")

    def onLogout(self, sessionID):
        print(f"FIX logout: {sessionID}")

    def toAdmin(self, message, sessionID):
        msg_type = fix.MsgType()
        message.getHeader().getField(msg_type)
        if msg_type.getValue() == fix.MsgType_Logon:
            # Saxo requires Username(553), Password(554), HeartBtInt(108), ResetSeqNumFlag(141) per session rules
            message.setField(fix.EncryptMethod(0))
            message.setField(fix.HeartBtInt(30))
            if self.creds.get('resetSeq', False):
                message.setField(fix.ResetSeqNumFlag(True))
            if 'username' in self.creds:
                message.setField(fix.Username(self.creds['username']))
            if 'password' in self.creds:
                message.setField(fix.Password(self.creds['password']))

    def fromAdmin(self, message, sessionID):
        pass

    def toApp(self, message, sessionID):
        pass

    def fromApp(self, message, sessionID):
        print(f"App message: {message}")
        # Handle ExecutionReport(8), OrderCancelReject(9), TradingSessionStatus(h) as needed

    def start(self):
        self.initiator = fix.SocketInitiator(self, self.storeFactory, self.settings, self.logFactory)
        self.initiator.start()

    def stop(self):
        if self.initiator:
            self.initiator.stop()

    def _session(self) -> fix.SessionID:
        if not self.sessions:
            raise RuntimeError("No FIX session available")
        return self.sessions[0]

    def _apply_symbology(self, msg: fix44.NewOrderSingle, symbology: Dict):
        # Symbology per Saxo: use either 55 Symbol (FOR, INDEX with 775=1) or 22/48 (ISIN/UIC/etc.)
        # symbology example dict: { 'SecurityType':'FOR', 'Symbol':'EUR/USD' } or { 'SecurityIDSource':100, 'SecurityID':20567846 }
        if symbology.get('SecurityType'):
            msg.setField(fix.SecurityType(symbology['SecurityType']))
        if symbology.get('BookingType') is not None:
            msg.setField(fix.BookingType(symbology['BookingType']))
        if symbology.get('Symbol') and not symbology.get('SecurityIDSource'):
            msg.setField(fix.Symbol(symbology['Symbol']))
        if symbology.get('SecurityIDSource') and symbology.get('SecurityID'):
            msg.setField(fix.SecurityIDSource(str(symbology['SecurityIDSource'])))
            msg.setField(fix.SecurityID(str(symbology['SecurityID'])))
        if symbology.get('ExDestination'):
            msg.setField(fix.ExDestination(symbology['ExDestination']))
        if symbology.get('Currency'):
            msg.setField(fix.Currency(symbology['Currency']))

    def _apply_parties(self, msg: fix.Message):
        if not self.parties:
            return
        msg.setField(fix.NoPartyIDs(len(self.parties)))
        group = fix44.Message().getGroup(0)  # placeholder, construct proper group
        # Proper Parties repeating group
        parties_group = fix44.Parties()
        # quickfix python does not expose Parties class; construct via generic group
        party_group = fix.Group(453, 448)
        for p in self.parties:
            g = fix.Group(453, 448)
            g.setField(fix.PartyID(p['id']))
            g.setField(fix.PartyIDSource('D'))
            g.setField(fix.PartyRole(int(p['role'])))
            msg.addGroup(g)

    def _base_new_order(self, side: str, qty: float, account: str) -> fix44.NewOrderSingle:
        msg = fix44.NewOrderSingle(
            fix.ClOrdID(f"CL-{fix.UtcTimeStamp()}"),
            fix.Side(1 if side.upper() == "BUY" else 2),
            fix.TransactTime(),
            fix.OrdType(fix.OrdType_MARKET)
        )
        msg.setField(fix.OrderQty(qty))
        msg.setField(fix.Account(account))
        # Parties component
        self._apply_parties(msg)
        return msg

    # Market order
    def send_market_order(self, side: str, qty: float, account: str, tif: str = "DAY", symbology: Optional[Dict] = None):
        sessionID = self._session()
        msg = self._base_new_order(side, qty, account)
        if symbology:
            self._apply_symbology(msg, symbology)
        msg.setField(fix.OrdType(fix.OrdType_MARKET))
        msg.setField(fix.TimeInForce(0 if tif == "DAY" else 1))
        fix.Session.sendToTarget(msg, sessionID)

    # Limit order
    def send_limit_order(self, side: str, qty: float, price: float, account: str, tif: str = "DAY", symbology: Optional[Dict] = None):
        sessionID = self._session()
        msg = self._base_new_order(side, qty, account)
        if symbology:
            self._apply_symbology(msg, symbology)
        msg.setField(fix.OrdType(fix.OrdType_LIMIT))
        msg.setField(fix.Price(price))
        msg.setField(fix.TimeInForce(0 if tif == "DAY" else 1))
        fix.Session.sendToTarget(msg, sessionID)

    # Stop market order
    def send_stop_order(self, side: str, qty: float, stop_price: float, account: str, tif: str = "DAY", symbology: Optional[Dict] = None):
        sessionID = self._session()
        msg = self._base_new_order(side, qty, account)
        if symbology:
            self._apply_symbology(msg, symbology)
        msg.setField(fix.OrdType(fix.OrdType_STOP))
        msg.setField(fix.StopPx(stop_price))
        msg.setField(fix.TimeInForce(0 if tif == "DAY" else 1))
        fix.Session.sendToTarget(msg, sessionID)

    # Stop-limit order
    def send_stop_limit_order(self, side: str, qty: float, stop_price: float, limit_price: float, account: str, tif: str = "DAY", symbology: Optional[Dict] = None):
        sessionID = self._session()
        msg = self._base_new_order(side, qty, account)
        if symbology:
            self._apply_symbology(msg, symbology)
        msg.setField(fix.OrdType(fix.OrdType_STOP_LIMIT))
        msg.setField(fix.StopPx(stop_price))
        msg.setField(fix.Price(limit_price))
        msg.setField(fix.TimeInForce(0 if tif == "DAY" else 1))
        fix.Session.sendToTarget(msg, sessionID)

    # IOC / FOK support via TimeInForce
    def send_limit_order_ioc(self, side: str, qty: float, price: float, account: str, symbology: Optional[Dict] = None):
        sessionID = self._session()
        msg = self._base_new_order(side, qty, account)
        if symbology:
            self._apply_symbology(msg, symbology)
        msg.setField(fix.OrdType(fix.OrdType_LIMIT))
        msg.setField(fix.Price(price))
        msg.setField(fix.TimeInForce(fix.TimeInForce_IMMEDIATE_OR_CANCEL))
        fix.Session.sendToTarget(msg, sessionID)

    def send_limit_order_fok(self, side: str, qty: float, price: float, account: str, symbology: Optional[Dict] = None):
        sessionID = self._session()
        msg = self._base_new_order(side, qty, account)
        if symbology:
            self._apply_symbology(msg, symbology)
        msg.setField(fix.OrdType(fix.OrdType_LIMIT))
        msg.setField(fix.Price(price))
        msg.setField(fix.TimeInForce(fix.TimeInForce_FILL_OR_KILL))
        fix.Session.sendToTarget(msg, sessionID)

    # Replace (OrderCancelReplaceRequest)
    def replace_order(self, orig_cl_ord_id: str, side: str, qty: float, account: str, tif: str, ord_type: int, new_price: Optional[float] = None, new_stop_px: Optional[float] = None):
        sessionID = self._session()
        msg = fix44.OrderCancelReplaceRequest(
            fix.OrigClOrdID(orig_cl_ord_id),
            fix.ClOrdID(f"CR-{fix.UtcTimeStamp()}"),
            fix.HandlInst(fix.HandlInst_MANUAL_ORDER_BEST_EXECUTION),
            fix.Side(1 if side.upper() == "BUY" else 2),
            fix.TransactTime()
        )
        msg.setField(fix.OrderQty(qty))
        msg.setField(fix.OrdType(ord_type))
        msg.setField(fix.TimeInForce(tif))
        if new_price is not None:
            msg.setField(fix.Price(new_price))
        if new_stop_px is not None:
            msg.setField(fix.StopPx(new_stop_px))
        msg.setField(fix.Account(account))
        self._apply_parties(msg)
        fix.Session.sendToTarget(msg, sessionID)

    # Cancel (OrderCancelRequest)
    def cancel_order(self, orig_cl_ord_id: str, account: str):
        sessionID = self._session()
        msg = fix44.OrderCancelRequest(
            fix.OrigClOrdID(orig_cl_ord_id),
            fix.ClOrdID(f"CX-{fix.UtcTimeStamp()}"),
            fix.Side(7),  # Saxo may return 54=7 in pending cancel if side unknown
            fix.TransactTime()
        )
        msg.setField(fix.Account(account))
        self._apply_parties(msg)
        fix.Session.sendToTarget(msg, sessionID)

    # Market Data Request (FX only per docs)
    def market_data_request_fx(self, mdreqid: str, symbol: str, depth: int = 1, subscribe: bool = True):
        sessionID = self._session()
        msg = fix44.MarketDataRequest(
            fix.MDReqID(mdreqid),
            fix.SubscriptionRequestType('1' if subscribe else '0'),
            fix.MarketDepth(depth)
        )
        # Entry types group: Bid and Offer
        no_types = fix.NoMDEntryTypes(2)
        msg.setField(no_types)
        g0 = fix.Group(267, 269)
        g0.setField(fix.MDEntryType(fix.MDEntryType_BID))
        msg.addGroup(g0)
        g1 = fix.Group(267, 269)
        g1.setField(fix.MDEntryType(fix.MDEntryType_OFFER))
        msg.addGroup(g1)
        # Related symbol group
        msg.setField(fix.NoRelatedSym(1))
        symGroup = fix.Group(146, 55)
        symGroup.setField(fix.Symbol(symbol))
        symGroup.setField(fix.SecurityType('FOR'))
        msg.addGroup(symGroup)
        fix.Session.sendToTarget(msg, sessionID)
